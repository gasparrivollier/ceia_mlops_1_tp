from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="XGBoost MLflow API", version="1.0")

# ----------------------------
# CONFIGURACIÓN CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir cualquier origen (ajustar en prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# CONFIG MLflow
# ----------------------------
mlflow.set_tracking_uri("http://mlflow:5000")
MODEL_URI = "models:/xgb_cantidad@latest"

# ----------------------------
# PATH BASELINE
# ----------------------------
BASELINE_PATH = "/opt/airflow/dags/data/baseline_por_dia_franja.parquet"

# ----------------------------
# CARGA MODELO
# ----------------------------
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print(f"Modelo cargado correctamente desde MLflow: {MODEL_URI}")
except Exception as e:
    print(f"Error cargando modelo desde MLflow: {e}")
    model = None

# ----------------------------
# CARGA BASELINE
# ----------------------------
try:
    baseline_df = pd.read_parquet(BASELINE_PATH)
    global_baseline_mean = baseline_df["baseline_cantidad"].mean()
    print(f"Baseline cargado correctamente desde: {BASELINE_PATH}")
except Exception as e:
    print(f"Error cargando baseline: {e}")
    baseline_df = None
    global_baseline_mean = None


# ----------------------------
# Validación del input
# ----------------------------
class InputData(BaseModel):
    """Features de entrada para el modelo."""
    dia: int
    franja: int
    barrio: int
    # ... agregar todos los features reales usados en el entrenamiento


# ----------------------------
# Función auxiliar: riesgo relativo
# ----------------------------
def compute_risk_metrics(dia: int, franja: int, pred_cantidad: float):
    """
    Calcula métricas de riesgo relativo usando el baseline por (dia, franja).

    Devuelve:
      - baseline_value: media histórica de 'cantidad' para ese (dia, franja)
      - relative_risk: pred / baseline (1 = riesgo promedio)
      - relative_risk_pct: % por encima/debajo del baseline
      - risk_score_0_1: índice 0-1 recortando en 3x el baseline
    """
    if baseline_df is None or global_baseline_mean is None:
        return None, None, None, None

    # Si hubieras hecho baseline por dia+franja+barrio, acá agregás barrio también
    row = baseline_df[
        (baseline_df["dia"] == dia) &
        (baseline_df["franja"] == franja)
    ]

    if not row.empty:
        baseline_value = float(row["baseline_cantidad"].iloc[0])
    else:
        # fallback a media global si no hay match
        baseline_value = float(global_baseline_mean)

    # evitar división por cero o valores raros
    if baseline_value <= 0:
        baseline_value = 1e-6

    relative_risk = pred_cantidad / baseline_value
    relative_risk_pct = (relative_risk - 1.0) * 100.0

    # índice 0–1: 0 = sin riesgo, 1 = 3x el baseline o más
    max_rr = 3.0
    risk_score_0_1 = min(relative_risk / max_rr, 1.0)
    risk_score_0_1 = float(max(risk_score_0_1, 0.0))

    return baseline_value, float(relative_risk), float(relative_risk_pct), risk_score_0_1


# ----------------------------
# ENDPOINT PREDICCIÓN
# ----------------------------
@app.post("/predict")
def predict(payload: InputData):

    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    # Convertir a dataframe
    df = pd.DataFrame([payload.dict()])

    try:
        df = df.astype({
            "dia": "int32",
            "franja": "float64",   # MLflow registró double → float64
            "barrio": "int32",
        })

        # Predicción de cantidad esperada
        pred = model.predict(df)
        pred_value = float(pred[0])

        # Calcular métricas de riesgo relativo usando el baseline
        baseline_value, relative_risk, relative_risk_pct, risk_score_0_1 = compute_risk_metrics(
            dia=int(df.loc[0, "dia"]),
            franja=int(df.loc[0, "franja"]),
            pred_cantidad=pred_value,
        )

        return {
            "prediction": pred_value,                 # cantidad esperada
            "baseline_cantidad": baseline_value,      # media histórica para ese dia/franja
            "relative_risk": relative_risk,           # ratio vs baseline (1 = riesgo promedio)
            "relative_risk_pct": relative_risk_pct,   # % sobre el baseline
            "risk_score_0_1": risk_score_0_1          # índice 0–1 para el front/mapa
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error haciendo inferencia: {e}")


# ----------------------------
# HEALTH CHECK
# ----------------------------
@app.get("/health")
def health():
    """
    Health check para que Kubernetes / Docker Compose validen disponibilidad.
    """
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "baseline_loaded": baseline_df is not None,
    }
