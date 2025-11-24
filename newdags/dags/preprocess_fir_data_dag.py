from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

import os
import pandas as pd
import re

RAW_DATA_PATH = "/opt/airflow/data/crimeData.csv"
CLEANED_DATA_PATH = "/opt/airflow/data/crimeData_cleaned.csv"


def normalize_malayalam(text):
    if pd.isna(text):
        return ""
    text = str(text)
    return text.strip()


def extract_act(sec_text):
    if pd.isna(sec_text):
        return ""
    sec_text = str(sec_text).strip()
    match = re.match(r"([A-Za-z]+)", sec_text)
    return match.group(1).upper() if match else ""


def extract_sections(sec_text):
    if pd.isna(sec_text):
        return []
    sec_text = str(sec_text)
    nums = re.findall(r"\d+", sec_text)
    return nums


def preprocess_fir_data():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw CSV not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    if "FIR_BRIEF" not in df.columns or "SEC_OF_LAW" not in df.columns:
        raise ValueError("Expected columns 'FIR_BRIEF' and 'SEC_OF_LAW' in raw CSV")

    df["FIR_BRIEF"] = df["FIR_BRIEF"].apply(normalize_malayalam)
    df["ACT"] = df["SEC_OF_LAW"].apply(extract_act)
    df["SECTIONS"] = df["SEC_OF_LAW"].apply(extract_sections)

    df = df[df["FIR_BRIEF"].str.len() > 3]

    os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Saved cleaned FIR data to {CLEANED_DATA_PATH}")


default_args = {
    "owner": "admin",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

with DAG(
    dag_id="preprocess_fir_data",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    preprocess_task = PythonOperator(
        task_id="preprocess_fir_data_task",
        python_callable=preprocess_fir_data,
    )
