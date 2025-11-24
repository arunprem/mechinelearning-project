from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "fir-section_predictor"   # NEW MODEL NAME

# Latest model version (created by Airflow DAG)
latest = client.get_latest_versions(model_name, stages=["None"])[0]
new_ver = latest.version
new_run_id = latest.run_id
new_acc = client.get_run(new_run_id).data.metrics.get("accuracy", 0)

# Current production model
prod_versions = client.get_latest_versions(model_name, stages=["Production"])
if prod_versions:
    prod_run_id = prod_versions[0].run_id
    prod_acc = client.get_run(prod_run_id).data.metrics.get("accuracy", 0)
else:
    prod_acc = 0

print(f"NEW_ACC={new_acc}, PROD_ACC={prod_acc}")

if new_acc > prod_acc:
    print("PROMOTE=YES")
else:
    print("PROMOTE=NO")
