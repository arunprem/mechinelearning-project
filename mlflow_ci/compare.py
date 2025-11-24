from mlflow.tracking import MlflowClient
import mlflow
import os
import sys
import pandas as pd

MODEL_NAME = "FIR-PREDICTOR-MODEL"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

print(f"🔍 Checking MLflow model registry for: {MODEL_NAME}")

# ---------------------------------------------------
# STEP 1 — Check if model exists in registry
# ---------------------------------------------------
registered = client.search_registered_models(filter_string=f"name = '{MODEL_NAME}'")

if len(registered) == 0:
    print(f"❌ Model '{MODEL_NAME}' not found in registry!")
    sys.exit(1)

print("✅ Model exists in registry.")

# ---------------------------------------------------
# STEP 2 — Get all versions of this model
# ---------------------------------------------------
versions = client.search_model_versions(f"name = '{MODEL_NAME}'")

production_version = None

for v in versions:
    if v.current_stage in ["Production"]:
        production_version = v

# If no production version
if production_version:
    prod_run_id = production_version.run_id
    prod_run = client.get_run(prod_run_id)
    prod_accuracy = float(prod_run.data.metrics.get("accuracy", 0))
    print(f"📘 Current Production version: {production_version.version}")
    print(f"📘 Production accuracy: {prod_accuracy}")
else:
    print("⚠️ No Production model found → promote new model.")
    prod_accuracy = -1   # anything will be higher than this

# ---------------------------------------------------
# STEP 3 — Get latest run accuracy
# ---------------------------------------------------
runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
latest_run_id = runs.iloc[0]["run_id"]
new_accuracy = float(runs.iloc[0]["metrics.accuracy"])

print(f"🆕 New model run_id: {latest_run_id}")
print(f"🆕 New model accuracy: {new_accuracy}")

decision = "PROMOTE" if new_accuracy > prod_accuracy else "KEEP_OLD"

with open("compare_result.txt", "w") as f:
    f.write(decision)

print(f"📌 Comparison result: {decision}")
