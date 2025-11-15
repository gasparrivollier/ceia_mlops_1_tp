from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
}

with DAG(
    dag_id="training_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",   
    catchup=False,
    default_args=default_args,
    tags=["modeling"],
) as dag:

    param_search = BashOperator(
        task_id="param_search",
        bash_command="python3 /opt/airflow/dags/scripts/hyperparam_search.py",
    )

    model_train = BashOperator(
        task_id="model_train",
        bash_command="python3 /opt/airflow/dags/scripts/model_train.py",
    )

    param_search >> model_train
