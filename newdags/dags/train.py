from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import mlflow
from mlflow import MlflowClient
import mlflow.pyfunc
import numpy as np
import os

# MLflow configuration
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MLFLOW_EXPERIMENT_NAME = "test_mlflow_cicd_integration"
MODEL_NAME = "fir-section_predictor"  # UPDATED MODEL NAME

# CSV file path (mounted in Airflow container)
CSV_PATH = "/opt/airflow/data/crimeData_cleaned.csv"

# MLflow setup
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def load_dataset(**context):
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Use only first 100 rows for training (as requested)
    df = df.head(100)

    context['ti'].xcom_push(key="raw_df", value=df.to_dict())
    print(f"Loaded {len(df)} rows for training")


def preprocess(**context):
    raw = context['ti'].xcom_pull(key='raw_df')
    df = pd.DataFrame(raw)

    df["brief"] = df["brief"].astype(str)
    df["act"] = df["act"].astype(str)
    df["section"] = df["section"].astype(str)

    # Simple feature engineering (example)
    df["features"] = df["brief"].apply(lambda x: len(x))

    context['ti'].xcom_push(key="clean_df", value=df.to_dict())
    print("Preprocessing completed")


def train_model(**context):
    df = pd.DataFrame(context['ti'].xcom_pull(key="clean_df"))

    # Dummy model training for testing CI/CD
    X = df["features"].values
    y = np.random.randint(0, 5, size=len(X))

    # Fake accuracy (simulate model improvement)
    accuracy = float(np.random.uniform(0.70, 0.95))

    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy", accuracy)

        # Simple dummy MLflow model
        class FIRModel(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input):
                return ["dummy_section_34"] * len(model_input)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=FIRModel(),
        )

        # Register in MLflow Model Registry
        client = MlflowClient()
        mv = client.create_model_version(
            name=MODEL_NAME,
            source=f"{run.info.artifact_uri}/model",
            run_id=run.info.run_id
        )

        print(f"NEW_MODEL_VERSION={mv.version}")


with DAG(
    dag_id="fir_model_train",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["ml", "training", "csv", "cicd"]
):

    task_load = PythonOperator(
        task_id="load_dataset",
        python_callable=load_dataset,
        provide_context=True
    )

    task_preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess,
        provide_context=True
    )

    task_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        provide_context=True
    )

task_load >> task_preprocess >> task_train
