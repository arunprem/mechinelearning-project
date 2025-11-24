from mlflow.tracking import MlflowClient
import mlflow
import os
import sys

MODEL_NAME = "fir-section_predictor"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# Read comparison decision
if not os.path.exists("compare_result.txt"):
    print("❌ compare_result.txt not found. Run compare.py first.")
    sys.exit(1)

decision = open("compare_result.txt").read().strip()

print(f"📌 Promotion decision: {decision}")

# ---- CASE 1: First model registered ----
if decision == "FIRST_MODEL_REGISTERED":
    print("🎉 First model registered. Marking it as Production...")
    latest = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest.version,
        stage="Production",
        archive_existing_versions=False
    )
    print(f"🚀 Model v{latest.version} promoted to PRODUCTION.")
    sys.exit(0)

# ---- CASE 2: Keep old model ----
if decision == "KEEP_OLD":
    print("ℹ️ Accuracy did not improve. Keeping existing production model.")
    sys.exit(0)

# ---- CASE 3: Promote new model ----
if decision == "PROMOTE":
    print("🚀 New model performs better. Promoting...")

    # Get best new model from latest run
    new_run = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
    new_run_id = new_run.iloc[0]["run_id"]

    # Find version corresponding to this run
    versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    new_version = None
    for v in versions:
        if v.run_id == new_run_id:
            new_version = v.version

    if new_version is None:
        print("❌ Could not find model version for the latest run!")
        sys.exit(1)

    # Promote to Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=new_version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"🎯 PROMOTED model version {new_version} to PRODUCTION")

else:
    print("❌ Unknown decision in compare_result.txt")
    sys.exit(1)
