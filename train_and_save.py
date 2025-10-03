import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
import joblib
from math import sqrt

# Configuration
DATA_PATH = "diabetes_dataset.csv"   # change if needed
OUT_DIR = "models"
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

for col in ['diagnosed_diabetes', 'diabetes_stage', 'hba1c']:
    if col not in df.columns:
        raise RuntimeError(f"Expected column '{col}' in dataset")

# Basic preprocessing
# Drop exact-duplicate patient rows 
df = df.drop_duplicates(subset='patient_id') if 'patient_id' in df.columns else df

# Handle missing values: small, simple strategy, for numeric columns: fill with median and for categorical columns: fill with mode
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# remove target numeric columns from "features" list
numeric_cols = [c for c in numeric_cols if c not in ('diagnosed_diabetes', 'hba1c')]

cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
# remove multiclass target from features
cat_cols = [c for c in cat_cols if c != 'diabetes_stage']

# Fill missing
for c in numeric_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "Unknown")

# Define features & targets
FEATURES = numeric_cols + cat_cols

X = df[FEATURES].copy()
y_binary = df['diagnosed_diabetes'].astype(int)
y_multi = df['diabetes_stage'].astype(str)   # keep as strings for LabelEncoder
y_reg = df['hba1c'].astype(float)

print("Number of features:", len(FEATURES))
print("Numeric:", numeric_cols)
print("Categorical:", cat_cols)

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', cat_transformer, cat_cols)
], remainder='drop')

# Fit preprocessor on full X (we will pipeline anyway)
preprocessor.fit(X)

# Save preprocessor for inference
joblib.dump(preprocessor, os.path.join(OUT_DIR, "preprocessor.pkl"))
print("Saved preprocessor ->", os.path.join(OUT_DIR, "preprocessor.pkl"))

# Transform features once to reduce repeated compute
X_transformed = preprocessor.transform(X)   # dense numpy array because sparse=False
print("X_transformed shape:", X_transformed.shape)

# Split indices for consistent train/test across tasks
train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary)

X_train = X_transformed[train_idx]
X_test = X_transformed[test_idx]

yb_train = y_binary.iloc[train_idx]
yb_test = y_binary.iloc[test_idx]

ym_train = y_multi.iloc[train_idx]
ym_test = y_multi.iloc[test_idx]

yr_train = y_reg.iloc[train_idx]
yr_test = y_reg.iloc[test_idx]

# Binary classification with XGBoost
print("\nTraining XGBoost binary classifier...")
xgb_bin = XGBClassifier(
    objective='binary:logistic',
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    verbosity=0
)

xgb_bin.fit(X_train, yb_train)
yb_pred = xgb_bin.predict(X_test)

print("Binary classification performance:")
print("Accuracy:", accuracy_score(yb_test, yb_pred))
print("Precision:", precision_score(yb_test, yb_pred))
print("Recall:", recall_score(yb_test, yb_pred))
print("F1:", f1_score(yb_test, yb_pred))
print(classification_report(yb_test, yb_pred))
joblib.dump(xgb_bin, os.path.join(OUT_DIR, "xgb_binary.pkl"))
print("Saved xgb_binary ->", os.path.join(OUT_DIR, "xgb_binary.pkl"))

# Multiclass classification with XGBoost
print("\nTraining XGBoost multiclass classifier...")
# encode labels
le_multi = LabelEncoder()
ym_train_enc = le_multi.fit_transform(ym_train)
ym_test_enc = le_multi.transform(ym_test)
joblib.dump(le_multi, os.path.join(OUT_DIR, "label_encoder_multiclass.pkl"))
print("Saved multiclass LabelEncoder ->", os.path.join(OUT_DIR, "label_encoder_multiclass.pkl"))
num_classes = len(le_multi.classes_)
print("Classes:", list(le_multi.classes_))

xgb_multi = XGBClassifier(
    objective='multi:softprob',
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    eval_metric='mlogloss',
    num_class=num_classes,
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    verbosity=0
)
xgb_multi.fit(X_train, ym_train_enc)
ym_pred_enc = xgb_multi.predict(X_test)
ym_pred = le_multi.inverse_transform(ym_pred_enc)

print("Multiclass classification performance:")
print("Accuracy:", accuracy_score(ym_test, ym_pred))
print(classification_report(ym_test, ym_pred))
joblib.dump(xgb_multi, os.path.join(OUT_DIR, "xgb_multi.pkl"))
print("Saved xgb_multi ->", os.path.join(OUT_DIR, "xgb_multi.pkl"))

# Regression (hba1c) with XGBoostRegressor
print("\nTraining XGBoost regressor for hba1c...")
xgb_reg = XGBRegressor(
    objective='reg:squarederror',
    random_state=RANDOM_STATE,
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    verbosity=0
)
xgb_reg.fit(X_train, yr_train)
yr_pred = xgb_reg.predict(X_test)

rmse = sqrt(mean_squared_error(yr_test, yr_pred))
mae = mean_absolute_error(yr_test, yr_pred)
r2 = r2_score(yr_test, yr_pred)
print(f"Regression (hba1c) performance: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

joblib.dump(xgb_reg, os.path.join(OUT_DIR, "xgb_reg.pkl"))
print("Saved xgb_reg ->", os.path.join(OUT_DIR, "xgb_reg.pkl"))

# Save metadata & feature names for inference
# Build feature names after preprocessing for interpretability
# numeric names + onehot feature names
num_names = numeric_cols
if cat_cols:
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    onehot_names = list(ohe.get_feature_names_out(cat_cols))
else:
    onehot_names = []
feature_names = num_names + onehot_names
joblib.dump(feature_names, os.path.join(OUT_DIR, "feature_names.pkl"))
print("Saved feature names ->", os.path.join(OUT_DIR, "feature_names.pkl"))

# Save also the preprocessor path already done
print("\nAll models & artifacts saved to:", OUT_DIR)
