from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

# Agregamos la carpeta de scripts al path de Python
sys.path.append("/opt/airflow/dags/scripts")

# Importamos la función que hace el trabajo
from preprocess import load_and_process

default_args = {
    "owner": "airflow",
}

with DAG(
    dag_id="preprocess_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",   
    catchup=False,
    default_args=default_args,
    tags=["preprocessing"],
) as dag:

    run_preprocess = PythonOperator(
        task_id="run_preprocess",
        python_callable=load_and_process,
    )
