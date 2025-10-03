# End-to-End-Diabetes-Risk-Prediction-Model
This project implements an end-to-end predictive machine learning pipeline for diabetes risk analysis using the Diabetes Health Indicators Dataset (100,000 records, 31 columns).

It covers:

Binary Classification → Predict whether a patient is diagnosed with diabetes (Yes/No).

Multiclass Classification → Predict the stage of diabetes (No Diabetes, Pre-Diabetes, Type 1, Type 2, Gestational).

Regression → Predict HbA1c (%) values.

Deployment → Interactive Streamlit Web App for predictions.

# Setup Instructions

1. Install dependencies

pip install -r requirements.txt

pip install pandas numpy scikit-learn xgboost streamlit joblib

2. Train the Models

python train_and_save.py

This will:

Preprocess dataset (scaling + one-hot encoding)

Train XGBoost models (binary, multiclass, regression)

Save all trained models and preprocessing artifacts under models/

Example training output:

Binary classification performance:
Accuracy: 0.844 | Precision: 0.90 | Recall: 0.83 | F1: 0.86

Multiclass classification performance:
Accuracy: 0.819 | Classes: ['Gestational', 'No Diabetes', 'Pre-Diabetes', 'Type 1', 'Type 2']

Regression performance:
RMSE = 0.251 | MAE = 0.201 | R2 = 0.906

3. Run Streamlit App

streamlit run app.py

Open in browser:

Local URL → http://localhost:8501 

Network URL → http://<your-ip>:8501 (for sharing on LAN)

# Features of Streamlit App

User-friendly input form (demographics, lifestyle, clinical measurements).

Predicts:

Diabetes status (Yes/No + probability).

Diabetes stage (multiclass).

HbA1c value (regression).

Interactive dashboard to test different profiles.

# Model Artifacts

Preprocessor → Handles scaling & one-hot encoding.

XGBoost Binary Classifier → Predicts diagnosed_diabetes.

XGBoost Multiclass Classifier → Predicts diabetes_stage.

XGBoost Regressor → Predicts hba1c.

Feature Names & Encoders → Stored for inference consistency.

# Future Improvements

Add SHAP/feature importance visualization.

Improve class balance (Gestational & Type 1 are underrepresented).

Deploy to cloud (Streamlit Cloud, Heroku, AWS, etc.).

# Disclaimer

This project is for educational/demo purposes only.

Not intended for clinical or medical decision-making without expert validation.

Author: Ittyavira C Abraham (MCA AI Student @ Amrita Vishwa Vidyapeetham)
