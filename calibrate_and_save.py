# calibrate_and_save.py
import pandas as pd
import numpy as np
import re
import pickle
import time
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# --- CONFIG ---
DATASET_PATH = "fake_job_postings.csv"
BASE_MODEL_PKL = "xgb_fake_job_detector_f1_0.8795_1765280759.pkl"  # your existing saved pipeline
TARGET_COL = "fraudulent"  # original column name in CSV (we will rename)

# --- Stopwords ---
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    print("ERROR: NLTK stopwords not found. Run: import nltk; nltk.download('stopwords')")
    STOPWORDS = set()

TEXT_COLS = ['title', 'company_profile', 'description', 'requirements', 'benefits']
CATEGORICAL_COLS = ['location', 'employment_type', 'required_experience',
                    'required_education', 'function', 'industry', 'department']
BINARY_COLS = ['telecommuting', 'has_company_logo', 'has_questions']

def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    text = ' '.join(w for w in text.split() if w not in STOPWORDS and len(w) > 1)
    return text

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure all required columns exist
    for c in TEXT_COLS + CATEGORICAL_COLS + BINARY_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Fill categorical/binary same as training
    for col in CATEGORICAL_COLS:
        df[col] = df[col].replace('', 'Missing').fillna('Missing')
    for col in BINARY_COLS:
        df[col] = df[col].fillna(0).astype(int)

    # Clean text
    for col in TEXT_COLS:
        df[col] = df[col].fillna('').apply(clean_text)

    # combined_text
    df['combined_text'] = df[TEXT_COLS].astype(str).agg(' '.join, axis=1)
    return df

def main():
    # 1) Load dataset
    df = pd.read_csv(DATASET_PATH)

    # Rename target for clarity
    df = df.rename(columns={TARGET_COL: 'is_fake'})
    df = df.drop(columns=['salary_range', 'job_id'], errors='ignore')

    y = df['is_fake']
    X = df.drop(columns=['is_fake'])

    # 2) Prepare features
    X = prepare_features(X)

    # 3) Load base model (Pipeline)
    with open(BASE_MODEL_PKL, "rb") as f:
        base_pipeline = pickle.load(f)

    # 4) Split: train_full / test, then train / calib
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    # 5) Fit base pipeline on train
    print("Fitting base pipeline on training split...")
    base_pipeline.fit(X_train, y_train)

    # 6) Calibrate probabilities on calibration split
    print("Calibrating probabilities (sigmoid)...")
    calibrated = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv="prefit"
    )
    calibrated.fit(X_calib, y_calib)

    # 7) Evaluate on test
    proba_test = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= 0.5).astype(int)

    f1 = f1_score(y_test, y_pred, pos_label=1)
    print("\n=== CALIBRATED MODEL TEST EVAL (threshold=0.5) ===")
    print(f"F1(fake=1): {f1:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # 8) Save calibrated model
    filename = f'calibrated_xgb_fake_job_detector_f1_{f1:.4f}_{int(time.time())}.pkl'
    with open(filename, "wb") as f:
        pickle.dump(calibrated, f)

    print(f"\n✅ Saved calibrated model to: {filename}")
    print("Tip: Update MODEL_FILENAME in app.py/streamlit_app.py to use this calibrated file.")

if __name__ == "__main__":
    main()
