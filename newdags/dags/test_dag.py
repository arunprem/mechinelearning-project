from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def hello():
    print("Test DAG CI/CD working!")

with DAG(
    dag_id="test_cicd",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False
):
    PythonOperator(
        task_id="hello_task",
        python_callable=hello
    )
