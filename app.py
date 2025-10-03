import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

MODEL_DIR = "models"

@st.cache_data
def load_artifacts():
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    xgb_bin = joblib.load(os.path.join(MODEL_DIR, "xgb_binary.pkl"))
    xgb_multi = joblib.load(os.path.join(MODEL_DIR, "xgb_multi.pkl"))
    le_multi = joblib.load(os.path.join(MODEL_DIR, "label_encoder_multiclass.pkl"))
    xgb_reg = joblib.load(os.path.join(MODEL_DIR, "xgb_reg.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    return preprocessor, xgb_bin, xgb_multi, le_multi, xgb_reg, feature_names

preprocessor, xgb_bin, xgb_multi, le_multi, xgb_reg, feature_names = load_artifacts()

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("Diabetes Risk Predictor — Demo")

st.markdown("""
This demo uses pre-trained XGBoost models for:
- Binary classification (diagnosed diabetes: Yes/No)
- Multiclass classification (diabetes stage)
- Regression (HBA1C prediction)
""")

# Build input form based on numeric & categorical fields saved in the preprocessor
# We infer feature template names from the preprocessor transformers:
num_cols = preprocessor.transformers_[0][2]
cat_cols = preprocessor.transformers_[1][2] if len(preprocessor.transformers_) > 1 else []

st.sidebar.header("Patient Information (inputs)")
inputs = {}

# Numeric inputs
for c in num_cols:
    # default value = median-ish (0) — user should fill realistically
    val = st.sidebar.number_input(f"{c}", value=float(0.0))
    inputs[c] = val

# Categorical inputs - present as selectbox of unique categories if possible
for c in cat_cols:
    # Try to load categories from the preprocessor OneHotEncoder categories_
    try:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        idx = list(cat_cols).index(c)
        categories = list(ohe.categories_[idx])
        default = categories[0] if categories else "Unknown"
        val = st.sidebar.selectbox(c, options=categories, index=0)
    except Exception:
        val = st.sidebar.text_input(c, value="")
    inputs[c] = val

if st.sidebar.button("Predict"):
    # Build DataFrame from inputs with correct column order
    X_input = pd.DataFrame([inputs])
    # Preprocess
    X_proc = preprocessor.transform(X_input)

    # Binary classification
    pred_bin = xgb_bin.predict(X_proc)[0]
    prob_bin = xgb_bin.predict_proba(X_proc)[0, 1] if hasattr(xgb_bin, "predict_proba") else None

    # Multiclass
    pred_multi_enc = xgb_multi.predict(X_proc)[0]
    pred_multi = le_multi.inverse_transform([int(pred_multi_enc)])[0]

    # Regression
    pred_hba1c = xgb_reg.predict(X_proc)[0]

    st.subheader("Results")
    st.write(f"**Diabetes (predicted)**: {'Yes' if pred_bin==1 else 'No'}")
    if prob_bin is not None:
        st.write(f"**Probability**: {prob_bin:.3f}")
    st.write(f"**Predicted diabetes stage**: {pred_multi}")
    st.write(f"**Predicted HbA1c**: {pred_hba1c:.3f}")

    st.info("This is a demo. For production use, do model monitoring, proper calibration, clinical validation, and security checks.")

st.markdown("---")
st.write("Model artifacts loaded from:", MODEL_DIR)
