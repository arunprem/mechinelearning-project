from mlflow.tracking import MlflowClient
import mlflow
import os
import sys
import pandas as pd
import requests

MODEL_NAME = "FIR-PREDICTOR-MODEL"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")

print("==============================================")
print(" MLflow CI/CD – Model Comparison Script")
print("==============================================")
print(f"MLFLOW_TRACKING_URI = {MLFLOW_TRACKING_URI}")
print(f"Model Name = {MODEL_NAME}")
print("Checking MLflow connection...")

# ----------------------------------------------------------
# STEP 0 — VERIFY MLflow SERVER IS REACHABLE
# ----------------------------------------------------------
try:
    r = requests.get(f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list", timeout=5)
    print(f"MLflow API Status Code: {r.status_code}")

    if r.status_code != 200:
        print("❌ MLflow server reachable but returned error response")
        sys.exit(1)

    print("✅ MLflow server is reachable.")
except Exception as e:
    print("❌ Cannot connect to MLflow server!")
    print(str(e))
    sys.exit(1)

# Create MLflow client
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# ----------------------------------------------------------
# STEP 1 — Ensure model exists
# ----------------------------------------------------------
print("\n🔍 Checking if model is registered...")

registered = client.search_registered_models(filter_string=f"name = '{MODEL_NAME}'")

if not registered:
    print(f"❌ Registered model '{MODEL_NAME}' not found!")
    sys.exit(1)

print("✅ Model is registered.")

# ----------------------------------------------------------
# STEP 2 — Check for production model
# ----------------------------------------------------------
versions = client.search_model_versions(f"name = '{MODEL_NAME}'")

production_version = None
for v in versions:
    if v.current_stage == "Production":
        production_version = v

if pro
