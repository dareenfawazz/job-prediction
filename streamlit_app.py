# streamlit_app.py (Updated: Threshold Slider + Logging + Calibration-Compatible + SHAP)

import streamlit as st
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
import numpy as np
import os
import csv
from datetime import datetime

import shap
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# --- Streamlit App Layout Setup (MUST BE FIRST COMMAND) ---
st.set_page_config(page_title="Job Fraud Detector (XAI)", layout="wide")

# --- Configuration ---
MODEL_FILENAME = 'calibrated_xgb_fake_job_detector_f1_0.8598_1765705186.pkl'
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "streamlit_predictions.csv")

# --- Preprocessing Constants ---
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    st.error("NLTK stopwords not found. Please run 'import nltk; nltk.download(\"stopwords\")' in a console.")
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

# --- SHAP Plotting Helper Function ---
def st_shap(plot, height=300):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

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
                "p_fake",
                "p_real",
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

def log_prediction(df_row: pd.Series, threshold: float, pred: int, p_fake: float, p_real: float):
    ensure_log_file()
    ts = datetime.utcnow().isoformat()

    def safe_len(v):
        try:
            return len(str(v)) if v is not None else 0
        except Exception:
            return 0

    label = "FAKE/FRAUDULENT" if pred == 1 else "REAL/LEGITIMATE"

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            ts,
            threshold,
            int(pred),
            label,
            float(p_fake),
            float(p_real),
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

# --- Model Loading and SHAP Initialization ---
@st.cache_resource
def load_predictor_and_explainability():
    model_path = os.path.join(os.getcwd(), MODEL_FILENAME)
    try:
        with open(model_path, 'rb') as file:
            loaded_obj = pickle.load(file)

        # We support:
        # 1) Pipeline (original)
        # 2) CalibratedClassifierCV (calibrated probabilities)
        if isinstance(loaded_obj, CalibratedClassifierCV):
            predictor = loaded_obj              # use this for predict_proba / predict
            base_pipeline = loaded_obj.estimator  # prefit pipeline for SHAP + transforms
        elif isinstance(loaded_obj, Pipeline):
            predictor = loaded_obj
            base_pipeline = loaded_obj
        else:
            st.error("Model file must be a scikit-learn Pipeline or CalibratedClassifierCV.")
            st.stop()

        # Extract components from the base pipeline (needed for SHAP)
        if not isinstance(base_pipeline, Pipeline):
            st.error("Internal error: base_pipeline is not a Pipeline.")
            st.stop()

        preprocessor = base_pipeline.steps[0][1]
        raw_model = base_pipeline.steps[-1][1]

        # SHAP Explainer for the raw XGBoost model
        explainer = shap.TreeExplainer(raw_model)

        return predictor, base_pipeline, explainer, raw_model, preprocessor

    except Exception as e:
        st.error(f"Error loading model or explainer: {e}")
        st.stop()

predictor, base_pipeline, explainer, raw_model, preprocessor = load_predictor_and_explainability()

# --- Prediction Function (Uses predictor for probabilities; base_pipeline for transforms) ---
def make_prediction(input_data):
    df_new = pd.DataFrame({k: [v] for k, v in input_data.items()})

    # Fill categorical/binary same as training
    for col in CATEGORICAL_COLS:
        df_new[col] = df_new[col].replace('', 'Missing').fillna('Missing')
    for col in BINARY_COLS:
        df_new[col] = df_new[col].fillna(0).astype(int)

    # Clean text + combined_text
    for col in TEXT_COLS:
        df_new[col] = df_new[col].fillna('').apply(clean_text)

    df_new['combined_text'] = df_new[TEXT_COLS].astype(str).agg(' '.join, axis=1)

    # Use predictor for calibrated probabilities (if available)
    proba = predictor.predict_proba(df_new)[0]
    p_fake = float(proba[1])
    p_real = float(proba[0])

    return df_new, p_fake, p_real

# --- UI ---
st.title("🤖 Job Fraud Detector Interface (Explainable AI)")
st.markdown("Enter the details of a job posting to get a prediction and see **why** the model made its decision.")

# Threshold control (sidebar)
st.sidebar.header("Decision Settings")
threshold = st.sidebar.slider(
    "Fraud threshold (classify as FAKE if P(fake) ≥ threshold)",
    min_value=0.0, max_value=1.0, value=0.5, step=0.01
)
st.sidebar.caption("Lower threshold → catch more fake jobs (higher recall), but more false alarms.")

with st.form("job_form"):
    st.subheader("Job Details & Metadata")

    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input("Job Title", "Remote Senior Python/Data Consultant")
        location = st.text_input("Location (e.g., London, UK)", "Anywhere, Global")
        department = st.text_input("Department (e.g., Engineering, Sales)", "")
        employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Other", "Missing"], index=2)
        required_experience = st.selectbox(
            "Experience Level",
            ["Mid-Senior level", "Entry level", "Director", "Executive", "Not Applicable", "Internship", "Missing"],
            index=0
        )

    with col2:
        required_education = st.selectbox(
            "Education Level",
            ["Bachelor's Degree", "Master's Degree", "High School or equivalent", "Doctorate", "Unspecified", "Missing"],
            index=0
        )
        industry = st.text_input("Industry (e.g., IT, Retail)", "Information Technology and Services")
        function = st.text_input("Function (e.g., Engineering, Marketing)", "Consulting")

        has_company_logo = st.selectbox("Has Company Logo?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", index=1)
        has_questions = st.selectbox("Has Screening Questions?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", index=1)
        telecommuting = st.selectbox("Is Telecommuting/Remote?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=1)

    st.subheader("Textual Information (Key for Prediction)")
    company_profile = st.text_area(
        "Company Profile",
        "Small independent firm providing short-term project consultation for data science needs. Established 3 years ago.",
        height=150
    )
    description = st.text_area(
        "Job Description",
        "Need expert Python consultant for 6-month contract. Project involves custom ETL pipelines and advanced data modeling.",
        height=200
    )
    requirements = st.text_area(
        "Requirements",
        "Proven ability to deliver projects independently. CV and sample work required. No phone calls.",
        height=100
    )
    benefits = st.text_area(
        "Benefits",
        "Hourly rate negotiable. Flexible work hours. No benefits package included (contract role).",
        height=100
    )

    submitted = st.form_submit_button("Analyze Posting")

if submitted:
    input_data = {
        'title': title,
        'location': location,
        'department': department,
        'company_profile': company_profile,
        'description': description,
        'requirements': requirements,
        'benefits': benefits,
        'employment_type': employment_type,
        'required_experience': required_experience,
        'required_education': required_education,
        'industry': industry,
        'function': function,
        'telecommuting': telecommuting,
        'has_company_logo': has_company_logo,
        'has_questions': has_questions,
    }

    df_processed, p_fake, p_real = make_prediction(input_data)

    # Decision using threshold on P(fake)
    pred = 1 if p_fake >= threshold else 0

    st.subheader("Prediction Result:")

    if pred == 1:
        st.error(f"🚨 WARNING: Classified as FAKE/FRAUDULENT (P(fake)={p_fake*100:.2f}%, threshold={threshold:.2f})")
    else:
        st.success(f"✅ Classified as REAL/LEGITIMATE (P(fake)={p_fake*100:.2f}%, threshold={threshold:.2f})")

    st.markdown("**Probability Meter (P(fake))**")
    st.progress(min(max(p_fake, 0.0), 1.0))

    st.info(f"Probability of REAL (0): {p_real:.4f}")
    st.warning(f"Probability of FAKE (1): {p_fake:.4f}")

    # Logging (do not log full texts; log lengths + key metadata)
    try:
        log_prediction(df_processed.iloc[0], threshold, pred, p_fake, p_real)
    except Exception as e:
        st.warning(f"Logging failed: {e}")

    # --- SHAP Explainability Plot ---
    st.subheader("🧐 Why did the model make this decision?")

    with st.spinner("Calculating SHAP values..."):
        # Transform for SHAP using the *base pipeline's* preprocessor
        X_numerical = preprocessor.transform(df_processed)

        raw_shap_output = explainer.shap_values(X_numerical)

        # class 1 (FAKE) values if list
        if isinstance(raw_shap_output, list) and len(raw_shap_output) == 2:
            shap_vals_for_plot = raw_shap_output[1]
            expected_value = explainer.expected_value[1]
        else:
            shap_vals_for_plot = raw_shap_output
            expected_value = explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1]

        X_numerical_dense = X_numerical.toarray()

        st_shap(shap.force_plot(
            expected_value,
            shap_vals_for_plot[0],
            X_numerical_dense[0]
        ))
