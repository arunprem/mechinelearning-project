from mlflow.tracking import MlflowClient
import mlflow
import json
import sys
import os

MODEL_NAME = "fir-section_predictor"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

print("🔍 Checking MLflow model registry...")

# -----------------------------------------
# Step 1 — Check if model exists (MLflow 3.x compatible)
# -----------------------------------------
registered = client.search_registered_models(filter_string=f"name = '{MODEL_NAME}'")

if len(registered) == 0:
    print(f"⚠️ No model named '{MODEL_NAME}' found in MLflow registry.")
    print("➡️ Registering FIRST model automatically...")

    runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        print("❌ No MLflow runs found! Run the Airflow training DAG first.")
        sys.exit(1)

    latest_run_id = runs.iloc[0]["run_id"]

    model_uri = f"runs:/{latest_run_id}/fir_prediction_model"
    result = mlflow.register_model(model_uri, MODEL_NAME)

    print(f"✅ First model registered with version: {result.version}")

    with open("compare_result.txt", "w") as f:
        f.write("FIRST_MODEL_REGISTERED")

    sys.exit(0)

# -----------------------------------------
# Step 2 — Compare new vs production
# -----------------------------------------
print("📘 Existing model found. Comparing accuracies...")

# MLflow 3.x: get latest versions
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

# Find production or staging version
prod_version = None
for v in versions:
    if v.current_stage in ["Production", "Staging"]:
        prod_version = v

if prod_version:
    prod_run_id = prod_version.run_id
    prod_run = client.get_run(prod_run_id)
    prod_acc = float(prod_run.data.metrics.get("accuracy", 0))
else:
    prod_acc = 0

print(f"Current Production/Staging accuracy: {prod_acc}")

# Latest run = new model training
new_run = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
new_run_id = new_run.iloc[0]["run_id"]
new_acc = float(new_run.iloc[0]["metrics.accuracy"])

print(f"New model accuracy: {new_acc}")

decision = "PROMOTE" if new_acc > prod_acc else "KEEP_OLD"

with open("compare_result.txt", "w") as f:
    f.write(decision)

print(f"📌 Comparison decision saved: {decision}")
