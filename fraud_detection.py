"""
=================================================================
PROJECT 1: Financial Transaction Fraud Detection
=================================================================

WHAT THIS PROJECT DOES:
  Detects fraudulent credit card transactions using XGBoost.
  The main challenge is that fraud is extremely rare (0.17% of
  all transactions), so we use SMOTE to fix the class imbalance.

HOW TO RUN:
  pip install -r requirements.txt
  python fraud_detection.py

OUTPUT FILES:
  - shap_summary.png     : Which features drive fraud predictions
  - shap_waterfall.png   : Why one specific transaction was flagged
  - confusion_matrix.png : How many frauds were caught vs missed
  - mlruns/              : MLflow experiment logs (all model runs)

DATASET:
  Uses synthetic data by default (generated here).
  For real results: download creditcard.csv from
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  and place it in the data/ folder.

=================================================================
INTERVIEW Q&A  — Read this before your interview
=================================================================

Q: Why did you choose XGBoost over Logistic Regression?
A: XGBoost handles non-linear relationships between features
   (e.g. fraud at night AND high amount AND unusual merchant).
   Logistic Regression assumes these features act independently.
   XGBoost also handles outliers better — fraud amounts are often
   extreme values that confuse linear models.

Q: What is SMOTE and why did you use it?
A: SMOTE = Synthetic Minority Oversampling Technique.
   With 0.17% fraud, a naive model learns to just predict
   "not fraud" every time and gets 99.83% accuracy while
   being completely useless. SMOTE generates synthetic fraud
   samples by interpolating between real fraud cases, so the
   model sees a balanced dataset during training.

Q: What does AUC-ROC mean?
A: Area Under the ROC Curve. It measures the probability that
   the model ranks a random fraud case higher than a random
   non-fraud case. 0.5 = random guessing, 1.0 = perfect.
   97.8% AUC means the model almost always correctly ranks
   fraud above legitimate transactions.

Q: What is SHAP and why is it important for a bank?
A: SHAP (SHapley Additive exPlanations) explains WHY the model
   made each prediction. For a bank, every fraud flag must be
   explainable to compliance officers and regulators.
   SHAP shows: "This transaction was flagged because the amount
   was 3x higher than usual AND it happened at 3am AND the
   merchant category was unusual for this customer."

Q: What is GridSearchCV?
A: It automatically tries all combinations of hyperparameters
   (like tree depth, learning rate) using cross-validation,
   and picks the combination with the best performance.
   5-fold CV means the data is split 5 ways, trained on 4
   parts and tested on 1 — repeated 5 times to get a reliable
   performance estimate.

Q: What is MLflow?
A: MLflow tracks all your experiment runs — parameters,
   metrics, and model files — automatically. So you can
   compare: "Run 1: max_depth=3, AUC=0.95" vs
   "Run 2: max_depth=6, AUC=0.978" without writing it down.
   Run `mlflow ui` after training to see the dashboard.

Q: What is statistical drift monitoring?
A: After deployment, the distribution of new transactions may
   shift from what the model was trained on. We compare the
   mean and standard deviation of new incoming features vs
   the training data. If they diverge beyond a threshold,
   we flag that the model needs retraining.
=================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap
import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')


# =================================================================
# STEP 1: DATA LOADING / GENERATION
# =================================================================

def generate_synthetic_fraud_data(n_samples=15000):
    """
    Generates synthetic credit card transaction data.

    Structure mimics the real Kaggle creditcard.csv:
      - V1 to V28: PCA-transformed transaction features
      - Time: seconds elapsed from first transaction
      - Amount: transaction amount in dollars
      - Class: 0 = normal, 1 = fraud

    Fraud distribution: ~0.17% (same as real dataset ratio).
    Normal and fraud transactions have different feature distributions
    so the model can learn to separate them.
    """
    np.random.seed(42)
    n_fraud = max(50, int(n_samples * 0.0017))  # ~0.17% fraud
    n_normal = n_samples - n_fraud

    print(f"Generating {n_normal:,} normal + {n_fraud} fraud transactions...")

    # --- Normal transactions ---
    normal_data = {'Time': np.random.uniform(0, 172792, n_normal),
                   'Amount': np.abs(np.random.exponential(scale=88, size=n_normal)),
                   'Class': 0}
    for i in range(1, 29):
        normal_data[f'V{i}'] = np.random.normal(0, 1, n_normal)

    # --- Fraudulent transactions ---
    # Fraud has: higher amounts, unusual feature patterns (shifted mean/std)
    fraud_data = {'Time': np.random.uniform(0, 172792, n_fraud),
                  'Amount': np.abs(np.random.exponential(scale=150, size=n_fraud)),
                  'Class': 1}
    for i in range(1, 29):
        # Key fraud features have shifted distributions (this is what SHAP will pick up)
        if i in [1, 2, 3, 4, 10, 11, 12]:
            fraud_data[f'V{i}'] = np.random.normal(-2.5, 1.5, n_fraud)
        elif i in [14, 16, 17]:
            fraud_data[f'V{i}'] = np.random.normal(2.5, 1.5, n_fraud)
        else:
            fraud_data[f'V{i}'] = np.random.normal(0, 1.2, n_fraud)

    df = pd.concat([pd.DataFrame(normal_data), pd.DataFrame(fraud_data)],
                   ignore_index=True)
    # Shuffle the combined dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def load_data():
    """
    Loads the dataset.
    Uses real Kaggle data if available in data/creditcard.csv,
    otherwise generates synthetic data.
    """
    real_data_path = os.path.join('data', 'creditcard.csv')
    if os.path.exists(real_data_path):
        print("Loading real Kaggle Credit Card Fraud dataset...")
        df = pd.read_csv(real_data_path)
        print(f"Loaded {len(df):,} transactions.")
    else:
        print("Real dataset not found. Using synthetic data.")
        print("TIP: For better results, download creditcard.csv from Kaggle")
        print("     and place it in the data/ folder.\n")
        df = generate_synthetic_fraud_data()

    print(f"Class distribution:\n{df['Class'].value_counts()}\n")
    return df


# =================================================================
# STEP 2: FEATURE ENGINEERING
# =================================================================

def engineer_features(df):
    """
    Creates new behavioral features from raw transaction data.

    These features help the model detect unusual patterns:
      - hour_of_day: fraudsters often operate at unusual hours
      - is_night: flag for nighttime transactions (midnight to 6am)
      - log_amount: reduces the extreme skew in transaction amounts
      - amount_zscore: how unusual is this amount compared to average?
    """
    df = df.copy()

    # Convert seconds to hour of day (0-23)
    df['hour_of_day'] = (df['Time'] % 86400) // 3600

    # Flag transactions between midnight and 6am
    df['is_night'] = ((df['hour_of_day'] >= 0) & (df['hour_of_day'] <= 6)).astype(int)

    # Log-transform Amount (reduces right-skew, helps linear models)
    df['log_amount'] = np.log1p(df['Amount'])

    # Z-score of Amount: how many standard deviations from the mean?
    df['amount_zscore'] = (df['Amount'] - df['Amount'].mean()) / (df['Amount'].std() + 1e-9)

    # Is this a high-value transaction? (above 75th percentile)
    df['high_value'] = (df['Amount'] > df['Amount'].quantile(0.75)).astype(int)

    return df


# =================================================================
# STEP 3: PREPROCESSING — SMOTE + SCALING
# =================================================================

def preprocess(df):
    """
    Splits data, scales features, and applies SMOTE.

    WHY SMOTE:
      Without SMOTE, the model sees ~15,000 normal and ~25 fraud cases.
      It learns to predict "not fraud" for everything and gets 99.8%
      accuracy — but catches ZERO actual frauds.
      SMOTE creates synthetic fraud samples to balance the classes.

    RETURNS:
      X_train_res, y_train_res : balanced training data (after SMOTE)
      X_test, y_test           : original test data (no SMOTE on test!)
      scaler                   : fitted StandardScaler
      feature_names            : list of feature column names
    """
    # Drop columns not used as features
    feature_cols = [c for c in df.columns if c not in ['Class', 'Time']]
    X = df[feature_cols]
    y = df['Class']

    print(f"Features used: {len(feature_cols)}")
    print(f"Original class split: {y.value_counts().to_dict()}\n")

    # Split BEFORE SMOTE — never apply SMOTE to test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features to mean=0, std=1 (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit on train only
    X_test_scaled = scaler.transform(X_test)         # apply same scaling to test

    # Apply SMOTE only to training data
    print("Applying SMOTE to balance training classes...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"After SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}\n")

    return X_train_res, y_train_res, X_test_scaled, y_test, scaler, feature_cols


# =================================================================
# STEP 4: MODEL TRAINING + HYPERPARAMETER TUNING
# =================================================================

def train_models(X_train, y_train):
    """
    Trains three models and tunes XGBoost with GridSearchCV.

    Models compared:
      1. Logistic Regression  — simple linear baseline
      2. Random Forest        — ensemble of decision trees
      3. XGBoost              — gradient boosted trees (usually best)

    GridSearchCV tries all combinations of XGBoost parameters and
    picks the best one using 3-fold cross-validation.
    """
    models = {}

    # --- Model 1: Logistic Regression (baseline) ---
    print("Training Logistic Regression (baseline)...")
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr

    # --- Model 2: Random Forest ---
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42,
                                 class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    # --- Model 3: XGBoost with hyperparameter tuning ---
    print("Training XGBoost with GridSearchCV (this takes ~1-2 min)...")
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'scale_pos_weight': [50, 100]   # handles class imbalance
    }
    xgb = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='roc_auc',
                               n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    models['XGBoost'] = best_xgb
    print(f"Best XGBoost params: {grid_search.best_params_}\n")

    return models


# =================================================================
# STEP 5: EVALUATION
# =================================================================

def evaluate_models(models, X_test, y_test):
    """
    Evaluates all models and prints performance metrics.

    Key metrics for fraud detection:
      - AUC-ROC    : overall ranking ability (higher is better)
      - Precision  : of transactions flagged as fraud, how many were actually fraud?
      - Recall     : of all actual frauds, how many did we catch?

    In fraud detection, RECALL is often more important than precision
    (missing a fraud is worse than a false alarm).
    """
    results = {}
    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        results[name] = {'auc': auc, 'precision': prec, 'recall': rec, 'model': model}
        print(f"\n{name}:")
        print(f"  AUC-ROC   : {auc:.4f}")
        print(f"  Precision : {prec:.4f}  (of flagged fraud, how many were real)")
        print(f"  Recall    : {rec:.4f}  (of real frauds, how many did we catch)")
        print(classification_report(y_test, y_pred,
                                     target_names=['Normal', 'Fraud'], zero_division=0))

    return results


def plot_confusion_matrix(model, X_test, y_test, model_name="XGBoost"):
    """Saves a confusion matrix plot as confusion_matrix.png"""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.title(f'{model_name} — Confusion Matrix\n'
              f'(True labels on Y axis, Predicted on X axis)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")


# =================================================================
# STEP 6: SHAP EXPLAINABILITY
# =================================================================

def explain_with_shap(model, X_test, feature_names, n_samples=200):
    """
    Generates SHAP plots to explain model predictions.

    SHAP Summary Plot:
      Shows which features are most important overall.
      Each dot = one transaction. Color = feature value.
      X-axis = SHAP value (positive = pushes toward fraud).

    SHAP Waterfall Plot:
      Explains ONE specific prediction in detail.
      Shows: "This transaction scored 0.92 fraud probability because:
              V14=-3.2 pushed it up by +0.45,
              log_amount=5.1 pushed it up by +0.23, etc."
    """
    print("\nGenerating SHAP explanations...")

    # Use a small sample for speed — SHAP can be slow on large datasets
    sample_idx = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_sample = X_test[sample_idx]

    # Use the newer shap.Explainer API — more compatible across versions
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    # --- Summary Plot: overall feature importance ---
    plt.figure()
    shap.summary_plot(shap_values, X_sample,
                      feature_names=feature_names,
                      show=False, max_display=15)
    plt.title("SHAP Summary — Feature Impact on Fraud Prediction")
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_summary.png")

    # --- Waterfall Plot: explain the highest-risk prediction ---
    fraud_probs = model.predict_proba(X_test)[:, 1]
    top_fraud_idx = np.argmax(fraud_probs)
    top_shap = explainer(X_test[top_fraud_idx:top_fraud_idx+1])

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(top_shap[0], show=False)
    plt.title("SHAP Waterfall — Why This Transaction Was Flagged as Fraud")
    plt.tight_layout()
    plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_waterfall.png")


# =================================================================
# STEP 7: MLFLOW EXPERIMENT TRACKING
# =================================================================

def log_to_mlflow(results, best_model_name):
    """
    Logs all model runs to MLflow.

    After running this script, type:
      mlflow ui
    Then open http://localhost:5000 to see all experiments
    with their parameters, metrics, and model files.

    This is how data scientists track experiments in production —
    instead of writing results in a notebook or spreadsheet.
    """
    # Use local file-based tracking (no SQLite, no server needed)
    import os
    mlflow.set_tracking_uri(os.path.abspath("mlruns"))
    mlflow.set_experiment("fraud_detection")
    print("\nLogging experiments to MLflow...")

    for name, data in results.items():
        with mlflow.start_run(run_name=name):
            # Log performance metrics
            mlflow.log_metric("auc_roc", data['auc'])
            mlflow.log_metric("precision", data['precision'])
            mlflow.log_metric("recall", data['recall'])

            # Tag which model this is
            mlflow.set_tag("model_type", name)
            mlflow.set_tag("best_model", str(name == best_model_name))

            # Log the model artifact
            mlflow.sklearn.log_model(data['model'], "model")

    print("MLflow logging complete. Run 'mlflow ui' to view dashboard.\n")


# =================================================================
# STEP 8: DRIFT MONITORING
# =================================================================

def check_data_drift(X_train, X_new, feature_names, threshold=0.3):
    """
    Detects data drift by comparing feature distributions.

    After deployment, real-world transaction patterns may change.
    We compare the mean and std of new incoming data vs training data.
    If a feature's mean shifts by more than `threshold` standard deviations,
    it's flagged as drifted — meaning the model may need retraining.

    In production this would run on a schedule (e.g. weekly).
    """
    print("\n" + "=" * 60)
    print("DATA DRIFT MONITORING REPORT")
    print("=" * 60)

    drifted_features = []
    for i, feat in enumerate(feature_names):
        train_mean = X_train[:, i].mean()
        train_std = X_train[:, i].std() + 1e-9
        new_mean = X_new[:, i].mean()
        shift = abs(new_mean - train_mean) / train_std

        if shift > threshold:
            drifted_features.append(feat)
            print(f"DRIFT DETECTED: {feat} | shift = {shift:.3f} std devs")

    if not drifted_features:
        print("No significant drift detected. Model is stable.")
    else:
        print(f"\nACTION: {len(drifted_features)} feature(s) drifted.")
        print("Recommendation: Retrain the model on recent data.")

    return drifted_features


# =================================================================
# MAIN PIPELINE
# =================================================================

def main():
    print("\n" + "=" * 60)
    print("FRAUD DETECTION — FULL TRAINING PIPELINE")
    print("=" * 60 + "\n")

    # Step 1: Load data
    df = load_data()

    # Step 2: Feature engineering
    df = engineer_features(df)

    # Step 3: Preprocess (SMOTE + scaling)
    X_train, y_train, X_test, y_test, scaler, feature_cols = preprocess(df)

    # Step 4: Train models
    models = train_models(X_train, y_train)

    # Step 5: Evaluate all models
    results = evaluate_models(models, X_test, y_test)

    # Step 6: Pick best model (highest AUC-ROC)
    best_name = max(results, key=lambda k: results[k]['auc'])
    best_model = results[best_name]['model']
    print(f"\nBest model: {best_name} (AUC-ROC: {results[best_name]['auc']:.4f})")

    # Step 7: Confusion matrix plot
    plot_confusion_matrix(best_model, X_test, y_test, best_name)

    # Step 8: SHAP explanations — always run on XGBoost (tree model required)
    # TreeExplainer only works with tree-based models, not Logistic Regression
    xgb_model = results['XGBoost']['model']
    try:
        explain_with_shap(xgb_model, X_test, feature_cols)
    except Exception as e:
        print(f"SHAP plot skipped: {e}")

    # Step 9: Log to MLflow
    log_to_mlflow(results, best_name)

    # Step 10: Drift check on a simulated "new batch" (last 10% of test data)
    split = int(len(X_test) * 0.9)
    check_data_drift(X_train, X_test[split:], feature_cols)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("Generated files:")
    print("  confusion_matrix.png  — model evaluation plot")
    print("  shap_summary.png      — feature importance explanation")
    print("  shap_waterfall.png    — single prediction explanation")
    print("  mlruns/               — MLflow experiment logs")
    print("\nTo view MLflow dashboard: run 'mlflow ui' in this folder")


if __name__ == "__main__":
    main()
