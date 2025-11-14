import os
import pandas as pd

# Path dentro del contenedor Airflow
DATA_DIR = "/opt/airflow/dataset"
OUTPUT_PATH = "/opt/airflow/dags/data/processed.parquet"

def load_and_process():

    # Detecta automáticamente todos los CSV del dataset
    csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    print(f"Se encontraron {len(csv_files)} archivos CSV.")

    df_list = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
            df_list.append(df)
            print(f"✔️  {os.path.basename(path)} ({len(df)} filas)")
        except Exception as e:
            print(f"⚠️ Error con {os.path.basename(path)}: {e}")

    # Concatena todo
    df = pd.concat(df_list, ignore_index=True)
    print("\nTotal filas combinadas:", len(df))

    # Ajustes
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df.dropna(inplace=True)

    df['anio'] = df['anio'].astype(int)
    df = df[~df['anio'].isin([2020, 2021])]   # Filtrar años

    # Drop columnas que no se usan
    df.drop(columns=['id-mapa', '_date', 'comuna', 'anio'], inplace=True, errors='ignore')

    # Crear dia_num, borrar fecha
    df['dia_num'] = df['fecha'].dt.day
    df.drop(columns=['fecha'], errors='ignore', inplace=True)

    # Categorical → encoded
    for col in ['dia', 'mes', 'uso_arma', 'uso_moto', 'tipo', 'barrio']:
        df[col] = df[col].astype('category').cat.codes

    df.drop(columns=['subtipo'], inplace=True, errors='ignore')

    # Agrupación de tu notebook
    df_grouped = (
        df.groupby(['dia', 'franja', 'barrio'])
          .agg({'cantidad':'sum'})
          .reset_index()
    )

    # Aseguramos directorio
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Export a parquet
    df_grouped.to_parquet(OUTPUT_PATH, index=False)
    print(f"✔️ Exportado a {OUTPUT_PATH}")

if __name__ == "__main__":
    load_and_process()
