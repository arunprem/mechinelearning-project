from mlflow.tracking import MlflowClient
import mlflow
import os
import sys

MODEL_NAME = "FIR-PREDICTOR-MODEL"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# Read comparison output
if not os.path.exists("compare_result.txt"):
    print("❌ ERROR: compare_result.txt missing.")
    sys.exit(1)

decision = open("compare_result.txt").read().strip()
print(f"📌 Promotion decision: {decision}")

if decision == "KEEP_OLD":
    print("ℹ️ Keeping existing production model.")
    sys.exit(0)

if decision != "PROMOTE":
    print("❌ Unknown decision value.")
    sys.exit(1)

print("🚀 PROMOTION STARTING — Selecting correct model version...")

# ---------------------------------------------------
# STEP 1 — Get model versions
# ---------------------------------------------------
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

# Get the most recently logged run
runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
latest_run_id = runs.iloc[0]["run_id"]

latest_version = None

# Find which model version matches this run
for v in versions:
    if v.run_id == latest_run_id:
        latest_version = v

if latest_version is None:
    print("❌ Could not locate model version for latest run.")
    sys.exit(1)

print(f"🎯 Promoting version {latest_version.version} to Production…")

# ---------------------------------------------------
# STEP 2 — Promote
# ---------------------------------------------------
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"🎉 SUCCESS — Model v{latest_version.version} is now PRODUCTION!")
