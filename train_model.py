"""
Credit Card Approval – Model Training Pipeline
================================================
Trains four classifiers on the UCI Credit Card Approval dataset:
  1. Logistic Regression
  2. Random Forest
  3. XGBoost (Gradient Boosting)
  4. Decision Tree

The best model (by ROC-AUC) is serialised to models/best_model.pkl together
with the fitted pre-processing pipeline (models/preprocessor.pkl) so that the
Flask app and the export script can load them without touching
the raw training data.

Dataset columns assumed (based on the anonymised UCI set):
  A1  – Gender            (categorical)
  A2  – Age               (continuous)
  A3  – Debt              (continuous)
  A4  – Married           (categorical)
  A5  – BankCustomer      (categorical)
  A6  – EducationLevel    (categorical)
  A7  – Ethnicity         (categorical)
  A8  – YearsEmployed     (continuous)
  A9  – PriorDefault      (binary categorical)
  A10 – Employed          (binary categorical)
  A11 – CreditScore       (continuous)
  A12 – DriversLicense    (binary categorical)
  A13 – Citizen           (categorical)
  A14 – ZipCode           (continuous  – treated as categorical)
  A15 – Income            (continuous)
  A16 – Approved          (target: + / -)
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  Load data
# ─────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "crx.data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

COLUMN_NAMES = [
    "Gender", "Age", "Debt", "Married", "BankCustomer",
    "EducationLevel", "Ethnicity", "YearsEmployed", "PriorDefault",
    "Employed", "CreditScore", "DriversLicense", "Citizen",
    "ZipCode", "Income", "Approved"
]

print("Loading dataset …")
df = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES, na_values="?")

print(f"  Shape: {df.shape}")
print(f"  Missing values:\n{df.isnull().sum()}\n")

# ─────────────────────────────────────────────
# 2.  Feature engineering
# ─────────────────────────────────────────────
# Encode target (+  → 1 ,  -  → 0)
df["Approved"] = df["Approved"].map({"+": 1, "-": 0})

# ZipCode: treat as categorical string
df["ZipCode"] = df["ZipCode"].astype(str)

# Split features / target
X = df.drop("Approved", axis=1)
y = df["Approved"]

# Identify column types
NUMERIC_COLS = ["Age", "Debt", "YearsEmployed", "CreditScore", "Income"]
CATEGORICAL_COLS = [
    "Gender", "Married", "BankCustomer", "EducationLevel",
    "Ethnicity", "PriorDefault", "Employed", "DriversLicense",
    "Citizen", "ZipCode"
]

# ─────────────────────────────────────────────
# 3.  Pre-processing pipeline
# ─────────────────────────────────────────────
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline,  NUMERIC_COLS),
    ("cat", categorical_pipeline, CATEGORICAL_COLS),
])

# ─────────────────────────────────────────────
# 4.  Train / test split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}   Test size: {len(X_test)}\n")

# ─────────────────────────────────────────────
# 5.  Define classifiers
# ─────────────────────────────────────────────
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost (GradientBoosting)": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
    ),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
}

# ─────────────────────────────────────────────
# 6.  Train, evaluate, track best model
# ─────────────────────────────────────────────
results = {}
best_name  = None
best_auc   = -1
best_model = None

for name, clf in classifiers.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cv_auc = cross_val_score(pipe, X_train, y_train,
                             cv=5, scoring="roc_auc").mean()

    results[name] = {"accuracy": acc, "roc_auc": auc, "cv_roc_auc": cv_auc}

    print("-" * 55)
    print(f"  {name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}  (CV mean: {cv_auc:.4f})")
    print(classification_report(y_test, y_pred,
                                target_names=["Rejected", "Approved"]))
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    if auc > best_auc:
        best_auc   = auc
        best_name  = name
        best_model = pipe

# ─────────────────────────────────────────────
# 7.  Save best model + column metadata
# ─────────────────────────────────────────────
model_path = os.path.join(MODELS_DIR, "best_model.pkl")
meta_path  = os.path.join(MODELS_DIR, "model_meta.pkl")

with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

meta = {
    "best_model_name": best_name,
    "best_roc_auc":    best_auc,
    "numeric_cols":    NUMERIC_COLS,
    "categorical_cols": CATEGORICAL_COLS,
    "column_order":    list(X.columns),
    "all_results":     results,
}
with open(meta_path, "wb") as f:
    pickle.dump(meta, f)

print("=" * 55)
print(f"  Best model : {best_name}")
print(f"  ROC-AUC    : {best_auc:.4f}")
print(f"  Saved to   : {model_path}")
print(f"  Metadata   : {meta_path}")

# ─────────────────────────────────────────────
# 8.  Summary table
# ─────────────────────────────────────────────
print("\n  -- Summary --")
summary_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
print(summary_df.to_string())

