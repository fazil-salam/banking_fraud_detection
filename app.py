"""
=================================================================
Project  : Financial Transaction Fraud Detection
File     : app.py
Author   : Abdul Fazil Abdul Salam
Date     : 2026-03-17
Version  : 1.0
Purpose  : Flask REST API that loads the trained XGBoost fraud model
           and serves real-time predictions via POST /predict.
           Applies live feature engineering using training-set
           statistics and returns fraud probability + risk level.
=================================================================

=================================================================
PROJECT 1: Financial Transaction Fraud Detection — Flask API
=================================================================

Loads the trained XGBoost model and exposes a REST API to score
live credit card transactions in real-time.

HOW TO RUN:
  1. Train the model first:
       python fraud_detection.py
  2. Start the API:
       python app.py
  3. API runs at http://localhost:5003

ENDPOINTS:
  GET  /health     → health check
  GET  /info       → model info and feature list
  POST /predict    → score a single transaction

EXAMPLE CURL COMMANDS:

  # Health check
  curl http://localhost:5003/health

  # Score a normal transaction
  curl -X POST http://localhost:5003/predict \
    -H "Content-Type: application/json" \
    -d '{
      "Time": 45003,
      "Amount": 25.50,
      "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
      "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
      "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
      "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
      "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
      "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
      "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02
    }'

  # Score a suspicious transaction (high-risk pattern)
  curl -X POST http://localhost:5003/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406,
    "Amount": 149.62,
    "V1": -1.36, "V2": -0.07, "V3": -3.04, "V4": -2.10,
    "V5": -0.31, "V6": -1.39, "V7": -0.05, "V8": 0.13,
    "V9": -0.21, "V10": -9.32, "V11": 2.11, "V12": -11.88,
    "V13": 0.14, "V14": -13.46, "V15": -0.69, "V16": -3.99,
    "V17": -7.72, "V18": -4.23, "V19": -3.24, "V20": -0.58,
    "V21": -0.17, "V22": -0.49, "V23": -0.03, "V24": 0.04,
    "V25": 0.37, "V26": 0.25, "V27": 0.10, "V28": 0.03
  }''

RISK LEVELS:
  LOW      : fraud_probability < 0.30
  MEDIUM   : fraud_probability 0.30 – 0.60
  HIGH     : fraud_probability 0.60 – 0.85
  CRITICAL : fraud_probability > 0.85
=================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Model paths ──────────────────────────────────────────────────
MODEL_DIR     = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH    = os.path.join(MODEL_DIR, 'fraud_model.pkl')
SCALER_PATH   = os.path.join(MODEL_DIR, 'scaler.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_cols.txt')
STATS_PATH    = os.path.join(MODEL_DIR, 'amount_stats.json')

# ── Load model artifacts at startup ─────────────────────────────
def load_artifacts():
    """Loads model, scaler, feature list, and Amount statistics."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run 'python fraud_detection.py' first to train and save the model."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH) as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    # Load training-set Amount statistics so single-transaction inference
    # computes amount_zscore and high_value using the correct population stats
    with open(STATS_PATH) as f:
        amount_stats = json.load(f)
    print(f"Model loaded  : {MODEL_PATH}")
    print(f"Features      : {len(feature_cols)} columns")
    print(f"Amount stats  : mean={amount_stats['mean']:.2f}  "
          f"std={amount_stats['std']:.2f}  q75={amount_stats['q75']:.2f}")
    return model, scaler, feature_cols, amount_stats

model, scaler, feature_cols, amount_stats = load_artifacts()


# ── Feature engineering (mirrors fraud_detection.py) ────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the same feature engineering used during training.

    IMPORTANT: amount_zscore and high_value use population-level
    statistics saved from the training dataset (amount_stats.json),
    NOT statistics computed from the single input row — that would
    produce NaN for std and wrong quantile for a single transaction.
    """
    df = df.copy()
    df['hour_of_day']   = (df['Time'] % 86400) // 3600
    df['is_night']      = ((df['hour_of_day'] >= 0) & (df['hour_of_day'] <= 6)).astype(int)
    df['log_amount']    = np.log1p(df['Amount'])
    # Use training-set statistics — consistent with how training features were built
    df['amount_zscore'] = (df['Amount'] - amount_stats['mean']) / (amount_stats['std'] + 1e-9)
    df['high_value']    = (df['Amount'] > amount_stats['q75']).astype(int)
    return df


def risk_level(prob: float) -> str:
    if prob < 0.30:
        return "LOW"
    elif prob < 0.60:
        return "MEDIUM"
    elif prob < 0.85:
        return "HIGH"
    else:
        return "CRITICAL"


# ── Routes ───────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model": "XGBoost Fraud Detector",
        "model_path": MODEL_PATH
    }), 200


@app.route('/info', methods=['GET'])
def info():
    """Returns model metadata and the list of required input features."""
    required_inputs = [c for c in ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]]
    return jsonify({
        "model"         : "XGBoost + SMOTE",
        "features_used" : feature_cols,
        "required_input": required_inputs,
        "engineered"    : ["hour_of_day", "is_night", "log_amount", "amount_zscore", "high_value"],
        "risk_levels"   : {
            "LOW"     : "fraud_probability < 0.30",
            "MEDIUM"  : "fraud_probability 0.30 – 0.60",
            "HIGH"    : "fraud_probability 0.60 – 0.85",
            "CRITICAL": "fraud_probability > 0.85"
        }
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Scores a single credit card transaction for fraud.

    Input  (JSON): Time, Amount, V1 … V28
    Output (JSON): fraud_probability, prediction, risk_level, engineered_features
    """
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    # Validate required fields
    required = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({
            "error"  : "Missing required fields",
            "missing": missing
        }), 400

    try:
        # Build DataFrame from input
        df = pd.DataFrame([data])

        # Apply same feature engineering as training
        df = engineer_features(df)

        # Select and order features exactly as during training
        X = df[feature_cols]

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        prob  = float(model.predict_proba(X_scaled)[0][1])
        label = "FRAUD" if prob >= 0.5 else "NORMAL"
        level = risk_level(prob)

        # Return engineered features for transparency
        engineered = {
            "hour_of_day"  : int(df['hour_of_day'].iloc[0]),
            "is_night"     : int(df['is_night'].iloc[0]),
            "log_amount"   : round(float(df['log_amount'].iloc[0]), 4),
            "amount_zscore": round(float(df['amount_zscore'].iloc[0]), 4),
            "high_value"   : int(df['high_value'].iloc[0]),
        }

        return jsonify({
            "fraud_probability" : round(prob, 6),
            "prediction"        : label,
            "risk_level"        : level,
            "engineered_features": engineered
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Entry point ──────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Fraud Detection API — starting on http://localhost:5003")
    print("=" * 60)
    print("  POST /predict  — score a transaction")
    print("  GET  /info     — model info + feature list")
    print("  GET  /health   — health check")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5003, debug=False)
