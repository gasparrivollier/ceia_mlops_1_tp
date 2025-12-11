from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
}

with DAG(
    dag_id="preprocess_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",   # corre una vez por día
    catchup=False,
    default_args=default_args,
    tags=["preprocessing"],
) as dag:

    # 1) Preprocesamiento: genera processed.parquet
    run_preprocess = BashOperator(
        task_id="run_preprocess",
        bash_command="python3 /opt/airflow/dags/scripts/preprocess.py",
    )

    # 2) Generar baseline a partir de processed.parquet
    generate_baseline = BashOperator(
        task_id="generate_baseline",
        bash_command="python3 /opt/airflow/dags/scripts/generate_baseline.py",
    )

    # Dependencias: primero preprocesar, luego baseline
    run_preprocess >> generate_baseline
