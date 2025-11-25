from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

import os
import json
import shutil

import pandas as pd
import numpy as np
import mlflow

import boto3

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# -------------------------------------------------------------------
# Paths & Config
# -------------------------------------------------------------------
DATA_PATH = "/opt/airflow/data/crimeData_cleaned.csv"

# We'll save the HF model directory, then archive it.
MODEL_DIR = "/opt/airflow/data/fir_bert_indic_model"
MODEL_ARCHIVE_PATH = "/opt/airflow/data/fir_prediction_model.tar.gz"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = os.getenv("MODEL_BUCKET", "fir-models")

HF_MODEL_NAME = "ai4bharat/indic-bert"  # IndicBERT base model


def train_fir_prediction_model():

    # ---------- Load dataset ----------
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cleaned data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # -----------------------------------------
    # USE UP TO 50,000 ROWS (for RAM safety)
    # -----------------------------------------
    TARGET_ROWS = 50000
    if len(df) > TARGET_ROWS:
        df = df.sample(TARGET_ROWS, random_state=42).reset_index(drop=True)

    print(f"Training using {len(df)} rows")

    # Ensure required columns exist
    required_cols = {"FIR_BRIEF", "ACT", "SEC_OF_LAW"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Combine ACT + SEC_OF_LAW into a single label string
    df["Target"] = df["ACT"].astype(str) + "||" + df["SEC_OF_LAW"].astype(str)

    # Keep only necessary columns
    df = df[["FIR_BRIEF", "Target"]].dropna()
    df["FIR_BRIEF"] = df["FIR_BRIEF"].astype(str)
    df["Target"] = df["Target"].astype(str)

    # ---------- Label Encoding ----------
    unique_labels = sorted(df["Target"].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df["label"] = df["Target"].map(label2id)

    num_labels = len(unique_labels)
    print(f"Number of unique labels (ACT||SEC): {num_labels}")

    # ---------- Train/Test Split ----------
    train_df, test_df = train_test_split(
        df, test_size=0.10, random_state=42, stratify=df["label"]
    )

    # Convert to Hugging Face Datasets
    train_ds = Dataset.from_pandas(train_df[["FIR_BRIEF", "label"]])
    test_ds = Dataset.from_pandas(test_df[["FIR_BRIEF", "label"]])

    # ---------- Tokenizer & Model ----------
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    def tokenize_function(batch):
        return tokenizer(
            batch["FIR_BRIEF"],
            padding="max_length",
            truncation=True,
            max_length=128,  # keep short for RAM/CPU
        )

    train_ds = train_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)

    # HF Trainer expects these columns
    train_ds = train_ds.remove_columns(["FIR_BRIEF"])
    test_ds = test_ds.remove_columns(["FIR_BRIEF"])
    train_ds.set_format("torch")
    test_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # OPTIONAL: Freeze some encoder layers to reduce RAM / speed-up
    # for name, param in model.named_parameters():
    #     if name.startswith("bert.embeddings") or "encoder.layer.0" in name:
    #         param.requires_grad = False

    # ---------- Metrics ----------
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "f1_macro": f1}

    # ---------- MLflow Setup ----------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("FIR-Act-Section-Prediction-IndicBERT")

    # ---------- Trainer Arguments ----------
    # Designed for 8GB RAM & (likely) CPU-only
    training_args = TrainingArguments(
        output_dir="/opt/airflow/data/fir_bert_indic_output",
        num_train_epochs=3,
        per_device_train_batch_size=8,    # reduce to 4 or 2 if OOM
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        fp16=False,                       # CPU training
        dataloader_num_workers=0,         # lower RAM pressure
    )

    with mlflow.start_run():

        # Log some params
        mlflow.log_params(
            {
                "hf_model_name": HF_MODEL_NAME,
                "num_labels": num_labels,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "max_length": 128,
                "batch_size": training_args.per_device_train_batch_size,
                "epochs": training_args.num_train_epochs,
            }
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        eval_metrics = trainer.evaluate()
        print("Evaluation metrics:", eval_metrics)

        # Log metrics to MLflow
        for k, v in eval_metrics.items():
            mlflow.log_metric(k, float(v))

        # ---------- Save model + tokenizer + label mapping locally ----------
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)

        mapping_path = os.path.join(MODEL_DIR, "label_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump({"label2id": label2id, "id2label": id2label}, f)

        # Log the entire directory as artifacts in MLflow
        mlflow.log_artifacts(MODEL_DIR, artifact_path="fir_prediction_model")

        # Create a tar.gz archive to upload to MinIO
        if os.path.exists(MODEL_ARCHIVE_PATH):
            os.remove(MODEL_ARCHIVE_PATH)

        # shutil.make_archive wants base_name *without* extension
        base_name = MODEL_ARCHIVE_PATH.replace(".tar.gz", "")
        shutil.make_archive(base_name, "gztar", MODEL_DIR)

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
            # Ignore if bucket already exists
            pass

        object_name = os.path.basename(MODEL_ARCHIVE_PATH)
        s3.upload_file(MODEL_ARCHIVE_PATH, BUCKET, object_name)
        print(f"Model archive uploaded to MinIO as {object_name}.")


# -------------------------------------------------------------------
# Airflow DAG
# -------------------------------------------------------------------
default_args = {
    "owner": "admin",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

with DAG(
    dag_id="fir_prediction_train_indicbert",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    train_task = PythonOperator(
        task_id="train_fir_prediction_model_task",
        python_callable=train_fir_prediction_model,
    )
