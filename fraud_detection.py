"""
=================================================================
Project  : Financial Transaction Fraud Detection
File     : fraud_detection.py
Author   : Abdul Fazil Abdul Salam
Date     : 2026-03-17
Version  : 1.0
Purpose  : Detects fraudulent credit card transactions on a highly
           imbalanced dataset (0.17% fraud) using XGBoost with SMOTE
           oversampling, SHAP explainability, MLflow experiment
           tracking, and automated data drift monitoring.
=================================================================

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
  - confusion_matrix.png : How many frauds were caught vs missed
  - roc_curve.png        : AUC-ROC curve comparing all three models
  - shap_summary.png     : Which features drive fraud predictions
  - shap_waterfall.png   : Why one specific transaction was flagged
  - mlruns/              : MLflow experiment logs (all model runs)
  - models/              : Saved model + scaler for Flask API

FLASK API:
  After training, run the REST API:
    python app.py
  Then POST to http://localhost:5000/predict to score live transactions.

DATASET:
  Uses synthetic data by default (generated here).
  For real results: download creditcard.csv from
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  and place it in the data/ folder.
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
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap
import joblib
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


def plot_roc_curve(results, X_test, y_test):
    """
    Plots ROC curves for all models on a single chart.

    ROC Curve shows the trade-off between:
      - True Positive Rate (Recall) on Y axis
      - False Positive Rate on X axis
    across every possible decision threshold.

    A perfect model hugs the top-left corner.
    The diagonal line = random guessing (AUC = 0.50).
    The shaded area under each curve = AUC-ROC score.

    Comparing all three models visually makes it clear
    why XGBoost was selected as the production model.
    """
    plt.figure(figsize=(9, 7))

    colors = {'Logistic Regression': '#e74c3c',
              'Random Forest':       '#2ecc71',
              'XGBoost':             '#2980b9'}

    for name, data in results.items():
        y_prob = data['model'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = data['auc']
        plt.plot(fpr, tpr,
                 label=f"{name}  (AUC = {auc:.4f})",
                 color=colors.get(name, 'gray'),
                 linewidth=2.5)

    # Diagonal baseline (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random baseline (AUC = 0.50)')

    plt.xlabel('False Positive Rate  (Normal transactions wrongly flagged)', fontsize=12)
    plt.ylabel('True Positive Rate  (Frauds correctly caught)', fontsize=12)
    plt.title('ROC Curve — Fraud Detection Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: roc_curve.png")


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

def log_to_mlflow(results, best_model_name, X_sample):
    """
    Logs all model runs to MLflow.

    After running this script, type:
      mlflow ui
    Then open http://localhost:5000 to see all experiments
    with their parameters, metrics, and model files.

    This is how data scientists track experiments in production —
    instead of writing results in a notebook or spreadsheet.

    X_sample: small array of input features used to infer model
    signature automatically — removes the MLflow signature warning.
    """
    mlflow.set_tracking_uri(os.path.abspath("mlruns"))
    mlflow.set_experiment("fraud_detection")
    print("\nLogging experiments to MLflow...")

    # Convert sample to DataFrame for clean signature inference
    X_sample_df = pd.DataFrame(X_sample[:5])

    for name, data in results.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_metric("auc_roc",   data['auc'])
            mlflow.log_metric("precision", data['precision'])
            mlflow.log_metric("recall",    data['recall'])
            mlflow.set_tag("model_type", name)
            mlflow.set_tag("best_model", str(name == best_model_name))

            # Use name= (not positional) and provide input_example to fix warnings
            mlflow.sklearn.log_model(
                sk_model=data['model'],
                name="model",
                input_example=X_sample_df
            )

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

    # Step 8: ROC curve — all three models on one chart
    plot_roc_curve(results, X_test, y_test)

    # Step 9: SHAP explanations — always run on XGBoost (tree model required)
    xgb_model = results['XGBoost']['model']
    try:
        explain_with_shap(xgb_model, X_test, feature_cols)
    except Exception as e:
        print(f"SHAP plot skipped: {e}")

    # Step 10: Save best model + scaler for Flask API
    import json
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/fraud_model.pkl')
    joblib.dump(scaler,     'models/scaler.pkl')
    # Save feature column names so API applies same engineering
    with open('models/feature_cols.txt', 'w') as f:
        f.write('\n'.join(feature_cols))
    # Save Amount statistics from training set so the Flask API can
    # compute amount_zscore and high_value correctly for single transactions
    # (cannot compute std/quantile from a single row — must use training stats)
    amount_stats = {
        'mean': float(df['Amount'].mean()),
        'std':  float(df['Amount'].std()),
        'q75':  float(df['Amount'].quantile(0.75))
    }
    with open('models/amount_stats.json', 'w') as f:
        json.dump(amount_stats, f, indent=2)
    print("Saved: models/fraud_model.pkl")
    print("Saved: models/scaler.pkl")
    print("Saved: models/feature_cols.txt")
    print("Saved: models/amount_stats.json  (Amount mean/std/q75 for API inference)")

    # Step 11: Log to MLflow (pass sample for clean signature inference)
    log_to_mlflow(results, best_name, X_train[:10])

    # Step 12: Drift check on a simulated "new batch" (last 10% of test data)
    split = int(len(X_test) * 0.9)
    check_data_drift(X_train, X_test[split:], feature_cols)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("Generated files:")
    print("  confusion_matrix.png  — model evaluation plot")
    print("  roc_curve.png         — AUC-ROC curve all models")
    print("  shap_summary.png      — feature importance explanation")
    print("  shap_waterfall.png    — single prediction explanation")
    print("  mlruns/               — MLflow experiment logs")
    print("  models/               — saved model + scaler for API")
    print("\nTo view MLflow dashboard: run 'mlflow ui' in this folder")
    print("To start Flask API:       run 'python app.py'")


if __name__ == "__main__":
    main()
