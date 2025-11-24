from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import boto3
import joblib

DATA_PATH = "/opt/airflow/data/crimeData_cleaned.csv"
MODEL_LOCAL_PATH = "/opt/airflow/data/fir_prediction_model.pkl"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = os.getenv("MODEL_BUCKET", "fir-models")


def train_fir_prediction_model():

    # ---------- Load dataset ----------
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cleaned data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # -----------------------------------------
    # RANDOM SAMPLE EXACTLY 5000 ROWS (Safe!)
    # -----------------------------------------
    TARGET_ROWS = 5000
    if len(df) > TARGET_ROWS:
        df = df.sample(TARGET_ROWS, random_state=42)

    print(f"Training using EXACTLY {len(df)} rows")

    # Ensure required columns exist
    required_cols = {"FIR_BRIEF", "ACT", "SEC_OF_LAW"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df["Target"] = df["ACT"].astype(str) + "||" + df["SEC_OF_LAW"].astype(str)

    X = df["FIR_BRIEF"].astype(str)
    y = df["Target"].astype(str)

    # ---------- Train/Test Split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42
    )

    # ---------- MLflow Setup ----------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("FIR-Act-Section-Prediction")

    # ---------- RAM Optimized TF-IDF + Logistic Regression ----------
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(1, 2),     # smaller for RAM
                    max_features=15000,     # reduced to safe limit
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=300,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    # ---------- Train + Log ----------
    with mlflow.start_run():

        pipeline.fit(X_train, y_train)
        accuracy = pipeline.score(X_test, y_test)

        print("Validation accuracy:", accuracy)
        mlflow.log_metric("accuracy", float(accuracy))

        # Log to MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="fir_prediction_model",
        )

        # Save model locally
        os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
        joblib.dump(pipeline, MODEL_LOCAL_PATH)

        # ---------- Upload to MinIO ----------
        endpoint_url = (
            f"http://{MINIO_ENDPOINT}"
            if not MINIO_ENDPOINT.startswith("http")
            else MINIO_ENDPOINT
        )

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
        )

        try:
            s3.create_bucket(Bucket=BUCKET)
        except Exception:
            pass

        s3.upload_file(MODEL_LOCAL_PATH, BUCKET, "fir_prediction_model.pkl")
        print("Model uploaded to MinIO.")


default_args = {
    "owner": "admin",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

with DAG(
    dag_id="fir_prediction_train",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    train_task = PythonOperator(
        task_id="train_fir_prediction_model_task",
        python_callable=train_fir_prediction_model,
    )
