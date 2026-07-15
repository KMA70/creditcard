"""
Credit Card Approval – Flask Web Application
=============================================
Loads the best pre-trained model (models/best_model.pkl) and serves
a web UI where users enter applicant details and receive an instant
approval / rejection decision.

Routes
------
  GET  /            → application form
  POST /predict     → prediction result page
  GET  /health      → JSON health-check (Watson ML / K8s probe)
  POST /api/predict → JSON prediction endpoint
"""

import os
import pickle

import pandas as pd
from flask import Flask, render_template, request, jsonify

# ── Bootstrap ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "model_meta.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

COLUMN_ORDER = meta["column_order"]
MODEL_NAME   = meta["best_model_name"]
MODEL_AUC    = meta["best_roc_auc"]
ALL_RESULTS  = meta["all_results"]


# ── Helper ────────────────────────────────────────────────────────────────────
def build_input_df(form: dict) -> pd.DataFrame:
    """Convert POST form / JSON dict into a one-row DataFrame that matches
    the column order expected by the sklearn Pipeline."""
    row = {
        "Gender":         form.get("gender", ""),
        "Age":            float(form.get("age", 0)),
        "Debt":           float(form.get("debt", 0)),
        "Married":        form.get("married", ""),
        "BankCustomer":   form.get("bank_customer", ""),
        "EducationLevel": form.get("education_level", ""),
        "Ethnicity":      form.get("ethnicity", ""),
        "YearsEmployed":  float(form.get("years_employed", 0)),
        "PriorDefault":   form.get("prior_default", ""),
        "Employed":       form.get("employed", ""),
        "CreditScore":    float(form.get("credit_score", 0)),
        "DriversLicense": form.get("drivers_license", ""),
        "Citizen":        form.get("citizen", ""),
        "ZipCode":        str(form.get("zip_code", "0")),
        "Income":         float(form.get("income", 0)),
    }
    return pd.DataFrame([row], columns=COLUMN_ORDER)


def confidence_label(prob: float) -> str:
    if prob >= 0.75:
        return "High"
    elif prob >= 0.55:
        return "Moderate"
    return "Low"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        model_name=MODEL_NAME,
        model_auc=f"{MODEL_AUC:.4f}",
        all_results=ALL_RESULTS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_df = build_input_df(request.form)
        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        approved      = bool(prediction == 1)
        prob_approved = float(probability[1])
        prob_rejected = float(probability[0])

        applicant = {
            "Age":            request.form.get("age"),
            "Income":         request.form.get("income"),
            "Debt":           request.form.get("debt"),
            "Credit Score":   request.form.get("credit_score"),
            "Years Employed": request.form.get("years_employed"),
            "Prior Default":  request.form.get("prior_default"),
            "Employed":       request.form.get("employed"),
        }

        return render_template(
            "result.html",
            approved=approved,
            prob_approved=f"{prob_approved * 100:.1f}",
            prob_rejected=f"{prob_rejected * 100:.1f}",
            confidence=confidence_label(prob_approved),
            model_name=MODEL_NAME,
            applicant=applicant,
        )
    except Exception as exc:
        return render_template("result.html", error=str(exc)), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "roc_auc": MODEL_AUC})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST JSON endpoint – same fields as the HTML form."""
    try:
        data        = request.get_json(force=True)
        input_df    = build_input_df(data)
        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        return jsonify({
            "approved":      bool(prediction == 1),
            "prob_approved": round(float(probability[1]), 4),
            "prob_rejected": round(float(probability[0]), 4),
            "model":         MODEL_NAME,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
