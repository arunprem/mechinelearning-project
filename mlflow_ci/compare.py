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
# Step 1 — Check if model exists
# -----------------------------------------
models = [m.name for m in client.list_registered_models()]

if MODEL_NAME not in models:
    print(f"⚠️ No model named '{MODEL_NAME}' found in MLflow registry.")
    print("➡️ Registering FIRST model automatically...")

    # Get last run
    runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        print("❌ No MLflow runs found! Train a model first.")
        sys.exit(1)

    latest_run_id = runs.iloc[0]["run_id"]

    # Register it
    model_uri = f"runs:/{latest_run_id}/fir_prediction_model"
    result = mlflow.register_model(model_uri, MODEL_NAME)

    print(f"✅ First model registered with version: {result.version}")

    # Save comparison result for Jenkins
    with open("compare_result.txt", "w") as f:
        f.write("FIRST_MODEL_REGISTERED")

    sys.exit(0)

# -----------------------------------------
# Step 2 — If model exists → compare accuracy
# -----------------------------------------
print("📘 Existing model found. Comparing accuracies...")

# Latest production or staging model
prod = client.get_latest_versions(MODEL_NAME, stages=["Production", "Staging", "None"])

if len(prod) == 0:
    print("⚠️ No existing model versions found. Registering the first one.")
    sys.exit(0)

prod_model = prod[0]
prod_run = client.get_run(prod_model.run_id)
prod_acc = float(prod_run.data.metrics.get("accuracy", 0))

print(f"Current production/staging accuracy: {prod_acc}")

# Get the last trained run
new_run = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
new_run_id = new_run.iloc[0]["run_id"]
new_acc = float(new_run.iloc[0]["metrics.accuracy"])

print(f"New model accuracy: {new_acc}")

decision = "PROMOTE" if new_acc > prod_acc else "KEEP_OLD"

with open("compare_result.txt", "w") as f:
    f.write(decision)

print(f"📌 Comparison result saved: {decision}")
