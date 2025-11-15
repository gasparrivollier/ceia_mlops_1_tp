from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="XGBoost MLflow API", version="1.0")

# ----------------------------
# CONFIG MLflow
# ----------------------------
mlflow.set_tracking_uri("http://mlflow:5000")

MODEL_URI = "models:/xgb_cantidad@latest"

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print(f"Modelo cargado correctamente desde MLflow: {MODEL_URI}")
except Exception as e:
    print(f"❌ Error cargando modelo desde MLflow: {e}")
    model = None


# ----------------------------
# Validación del input
# ----------------------------
class InputData(BaseModel):
    """Define automáticamente los features esperados."""
    # ⚠️ REEMPLAZAR con tus features reales:
    dia: int
    franja: int
    barrio: int
    # ... agregar todos los features


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
            "barrio": "int32"
        })
        pred = model.predict(df)
        return {"prediction": float(pred[0])}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error haciendo inferencia: {e}")


@app.get("/health")
def health():
    """
    Health check para que Kubernetes / Docker Compose validen disponibilidad.
    """
    return {"status": "ok", "model_loaded": model is not None}
