from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="XGBoost MLflow API", version="1.0")

# ----------------------------
# CONFIGURACIÓN CORS
# Esto se deberia ajustar en produccion, por ahora lo dejamos todo libre
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# CONFIG MLflow
# ----------------------------
mlflow.set_tracking_uri("http://mlflow:5000")

MODEL_URI = "models:/xgb_cantidad@latest"

model = None

def load_model():
    """
    Carga o recarga el modelo desde MLflow
    """
    global model
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print(f"✅ Modelo cargado correctamente desde MLflow: {MODEL_URI}")
        return True
    except Exception as e:
        print(f"❌ Error cargando modelo desde MLflow: {e}")
        model = None
        return False


# Cargar el modelo al iniciar la API
load_model()

# ----------------------------
# Validación del input
# ----------------------------
class InputData(BaseModel):
    """Define automáticamente los features esperados."""
    dia: int
    franja: int
    barrio: int


# ----------------------------
# ENDPOINT PREDICCIÓN
# ----------------------------
@app.post("/predict")
def predict(payload: InputData):

    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    df = pd.DataFrame([payload.dict()])

    try:
        df = df.astype({
            "dia": "int32",
            "franja": "float64",
            "barrio": "int32"
        })
        pred = model.predict(df)
        return {"prediction": float(pred[0])}

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error haciendo inferencia: {e}"
        )

# ----------------------------
# ENDPOINT HOT-RELOAD
# ----------------------------
@app.post("/reload-model")
def reload_model():
    """
    Recarga el modelo sin reiniciar el servicio
    """
    ok = load_model()
    if not ok:
        raise HTTPException(
            status_code=500,
            detail="No se pudo recargar el modelo desde MLflow."
        )
    return {"status": "ok", "message": "Modelo recargado correctamente"}

# ----------------------------
# HEALTHCHECK
# ----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }
