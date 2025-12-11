# /opt/airflow/dags/scripts/generate_baseline.py
import pandas as pd

DATA_PATH = "/opt/airflow/dags/data/processed.parquet"
BASELINE_PATH = "/opt/airflow/dags/data/baseline_por_dia_franja.parquet"

def main():
    df = pd.read_parquet(DATA_PATH)

    baseline = (
        df.groupby(["dia", "franja"])["cantidad"]
          .mean()
          .reset_index()
          .rename(columns={"cantidad": "baseline_cantidad"})
    )

    baseline.to_parquet(BASELINE_PATH)
    print("Baseline guardado en:", BASELINE_PATH)
