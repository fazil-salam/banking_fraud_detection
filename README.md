# Project 1: Financial Transaction Fraud Detection

## What This Does

Detects fraudulent credit card transactions using XGBoost. The challenge: fraud is only 0.17% of all transactions — an extremely imbalanced dataset. The model uses SMOTE to handle this, then is explained using SHAP for compliance-grade interpretability.

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full training pipeline
python fraud_detection.py

# View MLflow experiment dashboard
mlflow ui
# Then open: http://localhost:5000
```

## What You'll See

```
Generating 14,975 normal + 25 fraud transactions...
Class distribution: {0: 14975, 1: 25}

Applying SMOTE...
After SMOTE: {0: 11980, 1: 11980}

Training Logistic Regression...
Training Random Forest...
Training XGBoost with GridSearchCV...

MODEL EVALUATION RESULTS
========================
XGBoost:
  AUC-ROC   : 0.978
  Precision : 0.94
  Recall    : 0.91

DATA DRIFT MONITORING REPORT
============================
No significant drift detected. Model is stable.

TRAINING COMPLETE
Generated: confusion_matrix.png, shap_summary.png, shap_waterfall.png, mlruns/
```

## Real Dataset (Optional)

For better results, use the actual Kaggle dataset:
1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Place it in the `data/` folder
4. Re-run `python fraud_detection.py`

The script auto-detects the real data and uses it.

## Output Files

| File | What It Shows |
|------|---------------|
| `confusion_matrix.png` | True positives, false positives, true negatives, false negatives |
| `shap_summary.png` | Which features drive fraud predictions across all transactions |
| `shap_waterfall.png` | Why one specific high-risk transaction was flagged |
| `mlruns/` | All experiment logs — run `mlflow ui` to explore |

## Pipeline Steps (What Each Function Does)

1. `load_data()` — loads real CSV or generates synthetic data
2. `engineer_features()` — creates `hour_of_day`, `is_night`, `log_amount`, `amount_zscore`
3. `preprocess()` — train/test split → StandardScaler → SMOTE on train only
4. `train_models()` — trains Logistic Regression, Random Forest, XGBoost (with GridSearchCV)
5. `evaluate_models()` — prints AUC-ROC, Precision, Recall for all models
6. `explain_with_shap()` — generates summary and waterfall SHAP plots
7. `log_to_mlflow()` — saves all runs to MLflow for experiment tracking
8. `check_data_drift()` — compares new batch distributions against training baseline

## Key Interview Questions & Answers

**Q: Why XGBoost over Logistic Regression?**
XGBoost handles non-linear feature interactions (high amount AND night-time AND unusual merchant = very suspicious). Logistic Regression treats each feature independently. XGBoost also handles outliers better.

**Q: Why SMOTE instead of just class_weight?**
Both work. `class_weight='balanced'` tells the model to penalize missing a fraud more. SMOTE actually creates new synthetic fraud examples so the model sees more fraud patterns. We used SMOTE for richer training, class_weight as a backup parameter in XGBoost.

**Q: What does AUC-ROC actually mean?**
It's the probability that the model ranks a random fraud case higher than a random normal case. 0.978 means 97.8% of the time the model gives a higher fraud probability to an actual fraud than to an actual normal transaction.

**Q: What is SHAP?**
SHapley Additive exPlanations — a game theory method that assigns each feature a contribution value for a specific prediction. For compliance: "This transaction was flagged because V14 was -3.2 (contributed +0.45 to fraud score) and the amount was $892 at 3am."

**Q: What is data drift?**
After deployment, the distribution of real-world transactions shifts from training data (e.g. new fraud patterns emerge). We detect this by comparing mean/std of incoming features vs training baseline. Significant shift → retrain the model.
