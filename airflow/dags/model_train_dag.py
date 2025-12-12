from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

sys.path.append("/opt/airflow/dags/scripts")

from hyperparam_search import hyperparam_search as hyperparam_main
from model_train import model_train as model_train_main

default_args = {
    "owner": "airflow",
}

with DAG(
    dag_id="training_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["modeling"],
) as dag:

    param_search = PythonOperator(
        task_id="param_search",
        python_callable=hyperparam_main,
    )

    model_train = PythonOperator(
        task_id="model_train",
        python_callable=model_train_main,
    )

    param_search >> model_train
