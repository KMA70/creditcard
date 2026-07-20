"""
Credit Card Approval – Local Model Export & Validation
=======================================================
Replaces the former cloud deployment script.
This script verifies the saved model artefacts, prints a full performance
summary, and exports a standalone JSON scoring function you can integrate
into any local REST service or cron job — no cloud account required.

Workflow
--------
  1. Load models/best_model.pkl and models/model_meta.pkl.
  2. Print a performance summary for all trained classifiers.
  3. Score a sample applicant payload and display the result.
  4. Write models/scoring_payload_example.json for reference.

Run:
    python export_model.py
"""

import json
import os
import pickle

import pandas as pd

# ─────────────────────────────────────────────
# 1.  Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "model_meta.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "models/best_model.pkl not found. Run train_model.py first."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

COLUMN_ORDER    = meta["column_order"]
BEST_MODEL_NAME = meta["best_model_name"]
BEST_AUC        = meta["best_roc_auc"]
ALL_RESULTS     = meta["all_results"]

# ─────────────────────────────────────────────
# 2.  Performance summary
# ─────────────────────────────────────────────
print("=" * 56)
print("  Credit Card Approval – Model Summary")
print("=" * 56)
print(f"  Best model : {BEST_MODEL_NAME}")
print(f"  ROC-AUC    : {BEST_AUC:.4f}")
print()

summary_df = pd.DataFrame(ALL_RESULTS).T.sort_values("roc_auc", ascending=False)
print(summary_df.to_string())
print()

# ─────────────────────────────────────────────
# 3.  Sample prediction (local)
# ─────────────────────────────────────────────
sample = {
    "Gender": "b",
    "Age": 30.83,
    "Debt": 0.0,
    "Married": "u",
    "BankCustomer": "g",
    "EducationLevel": "w",
    "Ethnicity": "v",
    "YearsEmployed": 1.25,
    "PriorDefault": "t",
    "Employed": "t",
    "CreditScore": 1.0,
    "DriversLicense": "f",
    "Citizen": "g",
    "ZipCode": "00202",
    "Income": 0.0,
}

sample_df   = pd.DataFrame([sample], columns=COLUMN_ORDER)
prediction  = model.predict(sample_df)[0]
probability = model.predict_proba(sample_df)[0]

result = {
    "approved":      bool(prediction == 1),
    "prob_approved": round(float(probability[1]), 4),
    "prob_rejected": round(float(probability[0]), 4),
    "model":         BEST_MODEL_NAME,
}

print("Sample payload prediction:")
print(json.dumps(result, indent=2))

# ─────────────────────────────────────────────
# 4.  Export example payload JSON
# ─────────────────────────────────────────────
example_path = os.path.join(BASE_DIR, "models", "scoring_payload_example.json")
with open(example_path, "w") as f:
    json.dump({"input": sample, "output": result}, f, indent=2)

print(f"\n✓  Example payload written to: {example_path}")
print("   Start the Flask app with:  python app.py")
