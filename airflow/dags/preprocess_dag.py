from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
}

with DAG(
    dag_id="preprocess_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,   # ejecución manual
    catchup=False,
    default_args=default_args,
    tags=["preprocessing"],
) as dag:

    run_preprocess = BashOperator(
        task_id="run_preprocess",
        bash_command="python3 /opt/airflow/dags/preprocess.py",
    )
