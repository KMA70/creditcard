"""
IBM Watson Machine Learning – Deployment Pipeline
==================================================
This script demonstrates how to deploy the trained credit card approval
model to IBM Watson Machine Learning (WML) for scalable cloud inference.

Prerequisites
-------------
1. IBM Cloud account with Watson Machine Learning service provisioned.
2. WML credentials: API key, instance ID, and deployment space ID.
3. Best model + metadata saved locally by train_model.py.

Workflow
--------
  1. Authenticate with IBM Cloud using API key.
  2. Connect to Watson Machine Learning service.
  3. Save the pickled model artefact to WML model repository.
  4. Create an online deployment endpoint.
  5. Test the deployed endpoint with a sample payload.

Set these environment variables before running:
  WML_API_KEY        – IBM Cloud API key
  WML_INSTANCE_ID    – WML service instance GUID
  WML_SPACE_ID       – deployment space ID
  WML_LOCATION       – region (e.g. us-south, eu-de)
"""

import os
import pickle
import json

import pandas as pd
from ibm_watson_machine_learning import APIClient

# ─────────────────────────────────────────────
# 1.  Configuration
# ─────────────────────────────────────────────
API_KEY     = os.environ.get("WML_API_KEY")
INSTANCE_ID = os.environ.get("WML_INSTANCE_ID")
SPACE_ID    = os.environ.get("WML_SPACE_ID")
LOCATION    = os.environ.get("WML_LOCATION", "us-south")

if not all([API_KEY, INSTANCE_ID, SPACE_ID]):
    raise EnvironmentError(
        "Missing WML credentials. Set WML_API_KEY, WML_INSTANCE_ID, WML_SPACE_ID."
    )

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
META_PATH  = os.path.join(os.path.dirname(__file__), "models", "model_meta.pkl")

with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

MODEL_NAME  = meta["best_model_name"]
COLUMN_ORDER = meta["column_order"]

# ─────────────────────────────────────────────
# 2.  Connect to Watson ML
# ─────────────────────────────────────────────
wml_credentials = {
    "apikey":      API_KEY,
    "url":         f"https://{LOCATION}.ml.cloud.ibm.com",
    "instance_id": INSTANCE_ID,
}

client = APIClient(wml_credentials)
client.set.default_space(SPACE_ID)

print(f"✓  Connected to Watson ML ({LOCATION})")
print(f"   Space ID: {SPACE_ID}\n")

# ─────────────────────────────────────────────
# 3.  Store model in WML repository
# ─────────────────────────────────────────────
model_meta = {
    client.repository.ModelMetaNames.NAME: "CreditCardApprovalModel",
    client.repository.ModelMetaNames.TYPE: "scikit-learn_1.5",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: client.software_specifications.get_uid_by_name(
        "runtime-24.1-py3.11"
    ),
}

print("Uploading model to WML repository …")
stored_model = client.repository.store_model(
    model=MODEL_PATH,
    meta_props=model_meta,
)

model_uid = client.repository.get_model_id(stored_model)
print(f"✓  Model stored with UID: {model_uid}\n")

# ─────────────────────────────────────────────
# 4.  Create online deployment
# ─────────────────────────────────────────────
deployment_meta = {
    client.deployments.ConfigurationMetaNames.NAME: "CreditCardApproval_Deployment",
    client.deployments.ConfigurationMetaNames.ONLINE: {},
}

print("Creating online deployment …")
deployment = client.deployments.create(model_uid, meta_props=deployment_meta)
deployment_uid = client.deployments.get_uid(deployment)

print(f"✓  Deployment created with UID: {deployment_uid}")
print(f"   Status: {client.deployments.get_details(deployment_uid)['entity']['status']['state']}\n")

# ─────────────────────────────────────────────
# 5.  Test the deployed endpoint
# ─────────────────────────────────────────────
# Build a sample payload (must match the column order from training)
sample_payload = {
    "input_data": [
        {
            "fields": COLUMN_ORDER,
            "values": [
                [
                    "b",       # Gender
                    30.83,     # Age
                    0.0,       # Debt
                    "u",       # Married
                    "g",       # BankCustomer
                    "w",       # EducationLevel
                    "v",       # Ethnicity
                    1.25,      # YearsEmployed
                    "t",       # PriorDefault
                    "t",       # Employed
                    1,         # CreditScore
                    "f",       # DriversLicense
                    "g",       # Citizen
                    "00202",   # ZipCode
                    0,         # Income
                ]
            ],
        }
    ]
}

print("Testing deployed endpoint with sample payload …")
prediction = client.deployments.score(deployment_uid, sample_payload)

print(f"✓  Prediction result:\n{json.dumps(prediction, indent=2)}\n")

# ─────────────────────────────────────────────
# 6.  Summary
# ─────────────────────────────────────────────
print("═" * 60)
print("  Deployment Summary")
print("═" * 60)
print(f"  Model UID       : {model_uid}")
print(f"  Deployment UID  : {deployment_uid}")
print(f"  Endpoint URL    : {client.deployments.get_scoring_href(deployment)}")
print(f"  Space ID        : {SPACE_ID}")
print()
print("Use the scoring endpoint for real-time predictions from any client.")
print("Example curl:")
print(f"""
curl -X POST \\
  -H "Authorization: Bearer $IBM_CLOUD_IAM_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(sample_payload)}' \\
  {client.deployments.get_scoring_href(deployment)}
""")
