# app.py (Final Robust Version + Threshold + Logging + Calibration-Compatible)

import pickle
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import os
import csv
from datetime import datetime

# --- Configuration ---
MODEL_FILENAME = 'calibrated_xgb_fake_job_detector_f1_0.8598_1765705186.pkl'
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "api_predictions.csv")

# --- Preprocessing Constants ---
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    print("WARNING: NLTK stopwords not found. Run 'import nltk; nltk.download(\"stopwords\")' for best results.")
    STOPWORDS = set()

CATEGORICAL_COLS = [
    'location', 'employment_type', 'required_experience',
    'required_education', 'function', 'industry', 'department'
]
BINARY_COLS = ['telecommuting', 'has_company_logo', 'has_questions']
TEXT_COLS = ['title', 'company_profile', 'description', 'requirements', 'benefits']

# --- Helper Function for Text Cleaning ---
def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    text = ' '.join(word for word in text.split() if word not in STOPWORDS and len(word) > 1)
    return text

# --- Logging Helper ---
def ensure_log_file():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_utc",
                "threshold",
                "prediction_code",
                "label",
                "confidence_fake",
                "confidence_real",
                "location",
                "employment_type",
                "required_experience",
                "required_education",
                "industry",
                "function",
                "telecommuting",
                "has_company_logo",
                "has_questions",
                "title_len",
                "company_profile_len",
                "description_len",
                "requirements_len",
                "benefits_len",
            ])

def log_prediction(df_row: pd.Series, threshold: float, result: dict):
    ensure_log_file()
    ts = datetime.utcnow().isoformat()

    # Safely compute lengths (avoid logging full text content)
    def safe_len(v):
        try:
            return len(str(v)) if v is not None else 0
        except Exception:
            return 0

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            ts,
            threshold,
            result.get("prediction_code"),
            result.get("label"),
            result.get("confidence_fake"),
            result.get("confidence_real"),
            df_row.get("location", ""),
            df_row.get("employment_type", ""),
            df_row.get("required_experience", ""),
            df_row.get("required_education", ""),
            df_row.get("industry", ""),
            df_row.get("function", ""),
            df_row.get("telecommuting", 0),
            df_row.get("has_company_logo", 0),
            df_row.get("has_questions", 0),
            safe_len(df_row.get("title", "")),
            safe_len(df_row.get("company_profile", "")),
            safe_len(df_row.get("description", "")),
            safe_len(df_row.get("requirements", "")),
            safe_len(df_row.get("benefits", "")),
        ])

# --- Model Loading (Runs once on startup) ---
try:
    model_path = os.path.join(os.getcwd(), MODEL_FILENAME)
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print(f"✅ Model {MODEL_FILENAME} loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Model file {MODEL_FILENAME} not found. Check the path.")
    model = None

# --- Flask App Setup ---
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return """
    <html>
    <head><title>Job Fraud Detector API</title></head>
    <body>
        <h1>Job Fraud Detector API is Running!</h1>
        <p><strong>Endpoint:</strong> /predict</p>
        <p><strong>Method:</strong> POST</p>
        <p>Optional: provide <code>threshold</code> (0-1) in JSON to control decision boundary.</p>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server.'}), 500

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Invalid JSON format in request body.'}), 400

    # Optional threshold (default 0.5)
    threshold = 0.5
    if isinstance(data, dict) and "threshold" in data:
        try:
            threshold = float(data["threshold"])
        except Exception:
            return jsonify({'error': 'threshold must be a number between 0 and 1.'}), 400
        if not (0.0 <= threshold <= 1.0):
            return jsonify({'error': 'threshold must be between 0 and 1.'}), 400
        # remove threshold from features payload if present
        data = {k: v for k, v in data.items() if k != "threshold"}

    try:
        df_new = pd.DataFrame(data)
    except ValueError as e:
        return jsonify({'error': f'Invalid data format. Expected list values for each key in JSON: {e}'}), 400

    # Ensure required columns exist (helps with clearer errors)
    required_cols = set(CATEGORICAL_COLS + BINARY_COLS + TEXT_COLS)
    missing_cols = [c for c in required_cols if c not in df_new.columns]
    if missing_cols:
        return jsonify({'error': f'Missing required fields: {missing_cols}'}), 400

    # Preprocess
    for col in CATEGORICAL_COLS:
        df_new[col] = df_new[col].replace('', 'Missing').fillna('Missing')
    for col in BINARY_COLS:
        df_new[col] = df_new[col].fillna(0).astype(int)

    for col in TEXT_COLS:
        df_new[col] = df_new[col].fillna('').apply(clean_text)

    df_new['combined_text'] = df_new[TEXT_COLS].astype(str).agg(' '.join, axis=1)

    # Predict
    try:
        proba = model.predict_proba(df_new)[0]
        p_fake = float(proba[1])
        p_real = float(proba[0])

        # Decision by threshold on P(fake)
        pred = 1 if p_fake >= threshold else 0
    except Exception as e:
        return jsonify({'error': f'Prediction failed. Check input structure consistency: {e}'}), 500

    result = {
        'threshold': float(threshold),
        'prediction_code': int(pred),
        'label': 'FAKE/FRAUDULENT' if pred == 1 else 'REAL/LEGITIMATE',
        'confidence_fake': p_fake,
        'confidence_real': p_real
    }

    # Log (only first row)
    try:
        log_prediction(df_new.iloc[0], threshold, result)
    except Exception as e:
        # Don’t fail the API if logging fails
        print(f"WARNING: logging failed: {e}")

    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
