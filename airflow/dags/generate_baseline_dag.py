from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

sys.path.append("/opt/airflow/dags/scripts")

from preprocess import load_and_process as preprocess_main
from generate_baseline import main as baseline_main


default_args = {
    "owner": "airflow",
}

with DAG(
    dag_id="preprocess_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args=default_args,
    tags=["preprocessing"],
) as dag:

    run_preprocess = PythonOperator(
        task_id="run_preprocess",
        python_callable=preprocess_main,
    )

    generate_baseline = PythonOperator(
        task_id="generate_baseline",
        python_callable=baseline_main,
    )

    run_preprocess >> generate_baseline
