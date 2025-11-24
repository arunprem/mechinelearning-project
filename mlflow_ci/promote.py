from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "fir-section_predictor"

latest = client.get_latest_versions(model_name, stages=["None"])[0]

client.transition_model_version_stage(
    name=model_name,
    version=latest.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"Promoted version {latest.version} to Production")
