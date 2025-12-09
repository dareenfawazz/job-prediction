# app.py

import pickle
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import os # Added for better path handling

# --- Configuration (EDIT THIS) ---
MODEL_FILENAME = 'xgb_fake_job_detector_f1_0.8795_1765280759.pkl' # Use your exact filename

# --- Preprocessing Constants (MUST match training data exactly) ---
STOPWORDS = set(stopwords.words('english'))
CATEGORICAL_COLS = ['location', 'employment_type', 'required_experience', 
                    'required_education', 'function', 'industry', 'department']
BINARY_COLS = ['telecommuting', 'has_company_logo', 'has_questions']
TEXT_COLS = ['title', 'company_profile', 'description', 'requirements', 'benefits']


# --- Helper Function for Text Cleaning ---
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    # Fill NA/None values with empty string before joining, for safety
    text = ' '.join(word for word in text.split() if word not in STOPWORDS and len(word) > 1)
    return text


# --- Model Loading (Only runs once when the application starts) ---
try:
    model_path = os.path.join(os.getcwd(), MODEL_FILENAME)
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print(f"✅ Model {MODEL_FILENAME} loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Model file {MODEL_FILENAME} not found. Check the path.")
    model = None 


# --- Flask App Setup and Prediction Endpoint ---
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """Receives job posting data and returns a fraud prediction."""
    if model is None:
        return jsonify({'error': 'Model not loaded on server.'}), 500
        
    # 1. Get incoming JSON data
    data = request.get_json(force=True)
    
    # 2. Convert to DataFrame (Input values MUST be lists, e.g., {"title": ["New Title"]})
    try:
        df_new = pd.DataFrame(data)
    except ValueError as e:
        return jsonify({'error': f'Invalid data format. Expected list values in JSON: {e}'}), 400


    # 3. Apply the Same Preprocessing Pipeline as Training (CRITICAL)
    # 3a. Impute Categorical and Binary NaNs
    for col in CATEGORICAL_COLS:
        df_new[col] = df_new[col].fillna('Missing')
    for col in BINARY_COLS:
        df_new[col] = df_new[col].fillna(0).astype(int) 

    # 3b. Clean and Combine Text Features
    for col in TEXT_COLS:
        # Fill NaNs with empty string before cleaning if they slipped through
        df_new[col] = df_new[col].fillna('').apply(clean_text) 
        
    df_new['combined_text'] = df_new[TEXT_COLS].astype(str).agg(' '.join, axis=1)

    # 4. Make Prediction
    prediction = model.predict(df_new)[0]
    probability = model.predict_proba(df_new)[0]
    
    # 5. Format and Return Result
    result = {
        'prediction_code': int(prediction), # 0 (Real) or 1 (Fake)
        'label': 'FAKE/FRAUDULENT' if prediction == 1 else 'REAL/LEGITIMATE',
        'confidence_fake': float(probability[1]),
        'confidence_real': float(probability[0])
    }
    
    return jsonify(result)

@app.route('/', methods=['GET'])
def home():
    """Provides a simple home page and instructions for the API."""
    return """
    <html>
    <head><title>Job Fraud Detector API</title></head>
    <body>
        <h1>Job Fraud Detector API is Running!</h1>
        <p>This server is ready to receive prediction requests.</p>
        <p><strong>Endpoint:</strong> /predict</p>
        <p><strong>Method:</strong> POST</p>
        <p>To use the model, send a POST request with the job posting data in JSON format 
           to <code>http://127.0.0.1:5000/predict</code>.</p>
        <p>You cannot use this endpoint directly in your browser's address bar.</p>
    </body>
    </html>
    """

# --- Run the App ---
if __name__ == '__main__':
    # Flask runs locally on port 5000. Use debug=True for development.
    app.run(port=5000, debug=True)