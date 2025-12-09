# streamlit_app.py

import streamlit as st
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
import numpy as np
import os
import time 

# --- Streamlit App Layout Setup (MUST BE FIRST COMMAND) ---
st.set_page_config(page_title="Job Fraud Detector", layout="wide") 

# --- Configuration (MUST MATCH training and Flask app) ---
# Check your exact model filename and update it
MODEL_FILENAME = 'xgb_fake_job_detector_f1_0.8795_1765280759.pkl' 

# --- Preprocessing Constants ---
# NOTE: Ensure you have downloaded nltk stopwords if you haven't already.
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    # This should generally be handled before running the app, but included for robustness
    st.error("NLTK stopwords not found. Please run 'import nltk; nltk.download(\"stopwords\")' in a console.")
    STOPWORDS = set() 

CATEGORICAL_COLS = ['location', 'employment_type', 'required_experience', 
                    'required_education', 'function', 'industry', 'department']
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

# --- Model Loading (Uses Streamlit's cache to load only once) ---
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), MODEL_FILENAME)
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_FILENAME}' not found in the current directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# --- Prediction Function ---
def make_prediction(input_data):
    # 1. Convert input dictionary to a DataFrame
    # Note: Wrap scalar values in lists for DataFrame creation
    df_new = pd.DataFrame({k: [v] for k, v in input_data.items()})

    # 2. Apply Preprocessing Pipeline (Critical for Consistency)
    
    # 2a. Impute Categorical and Binary NaNs (Handling empty strings from text input)
    for col in CATEGORICAL_COLS:
        # If user leaves blank (''), replace with 'Missing'
        df_new[col] = df_new[col].replace('', 'Missing').fillna('Missing')
    
    # Binary columns are handled by st.selectbox, ensuring they are 0 or 1
    
    # 2b. Clean and Combine Text Features
    for col in TEXT_COLS:
        df_new[col] = df_new[col].fillna('').apply(clean_text) 
        
    df_new['combined_text'] = df_new[TEXT_COLS].astype(str).agg(' '.join, axis=1)

    # 3. Make Prediction
    prediction = model.predict(df_new)[0]
    probability = model.predict_proba(df_new)[0]
    
    return prediction, probability

# --- Streamlit App Layout ---

st.title("🤖 Job Fraud Detector Interface")
st.markdown("Enter the details of a job posting below to determine if it is **Legitimate** or **Fraudulent**.")

with st.form("job_form"):
    st.subheader("Job Details & Metadata")
    
    col1, col2 = st.columns(2)
    
    with col1:
        title = st.text_input("Job Title", "Senior Software Engineer")
        location = st.text_input("Location (e.g., London, UK)", "London, UK")
        department = st.text_input("Department (e.g., Engineering, Sales)", "")
        employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Other", "Missing"], index=0)
        required_experience = st.selectbox("Experience Level", ["Mid-Senior level", "Entry level", "Director", "Executive", "Not Applicable", "Internship", "Missing"], index=0)
        
    with col2:
        required_education = st.selectbox("Education Level", ["Bachelor's Degree", "Master's Degree", "High School or equivalent", "Doctorate", "Unspecified", "Missing"], index=0)
        industry = st.text_input("Industry (e.g., IT, Retail)", "Information Technology and Services")
        function = st.text_input("Function (e.g., Engineering, Marketing)", "Engineering")
        
        has_company_logo = st.selectbox("Has Company Logo?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        has_questions = st.selectbox("Has Screening Questions?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        telecommuting = st.selectbox("Is Telecommuting/Remote?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
    st.subheader("Textual Information (Key for Prediction)")
    company_profile = st.text_area("Company Profile", "A well-funded startup focused on AI solutions. (Leave blank if unknown)", height=150)
    description = st.text_area("Job Description", "Seeking experienced Python developer for cloud services. Must have Kubernetes knowledge.", height=200)
    requirements = st.text_area("Requirements", "5+ years professional experience. Excellent communication skills.", height=100)
    benefits = st.text_area("Benefits", "Healthcare, 401k match, unlimited PTO.", height=100)
    
    submitted = st.form_submit_button("Analyze Posting")

if submitted:
    # Compile inputs into the dictionary structure needed by the function
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
    
    # Run prediction
    prediction, probability = make_prediction(input_data)
    
    # --- Display Results ---
    
    st.subheader("Prediction Result:")
    
    # --- Corrected logic for confidence assignment ---

    if prediction == 1:
        st.error(f"🚨 WARNING: This posting is classified as FAKE/FRAUDULENT! (Confidence: {probability[1]*100:.2f}%)")
        # FIX: Convert NumPy float32 to standard Python float
        confidence_level = float(probability[1]) 
    else:
        st.success(f"✅ This posting is classified as REAL/LEGITIMATE. (Confidence: {probability[0]*100:.2f}%)")
        # FIX: Convert NumPy float32 to standard Python float
        confidence_level = float(probability[0]) 

    st.markdown(f"**Confidence Level:**")
    # This line will now work correctly
    st.progress(confidence_level)
    
    st.info(f"Probability of being REAL (0): {probability[0]:.4f}")
    st.warning(f"Probability of being FAKE (1): {probability[1]:.4f}")