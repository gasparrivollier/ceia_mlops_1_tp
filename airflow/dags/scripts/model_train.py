# ------------------------------
#  IMPORTS
# ------------------------------
import os
import pandas as pd
import json
import joblib
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from xgboost import DMatrix, train

# ------------------------------
#  PATHS
# ------------------------------
DATA_PATH = "/opt/airflow/dags/data/processed.parquet"
MODEL_PARAMS_PATH = "/opt/airflow/dags/models/best_model_params.json"
EVALS_PATH = "/opt/airflow/dags/models/best_model_evals.json"

# (opcional: copiar modelo local)
MODEL_PATH = "/opt/airflow/dags/models/best_model.xgb"

# ------------------------------
#  MLFLOW CONFIG
# ------------------------------
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("xgboost_training")


# ------------------------------
#  TRAIN FUNCTION
# ------------------------------
def model_train():

    # La variable de entorno ML_DEVICE debe indicar 'gpu' o 'cpu'.
    ml_device = os.environ.get("ML_DEVICE")
    if ml_device is None:
        raise RuntimeError("La variable de entorno ML_DEVICE no está definida. Configura ML_DEVICE=gpu o ML_DEVICE=cpu en el archivo .env")
    ml_device = ml_device.strip().lower()
    if ml_device not in ("gpu", "cpu"):
        raise RuntimeError("Valor inválido para ML_DEVICE. Usa 'gpu' o 'cpu'.")

    df = pd.read_parquet(DATA_PATH)

    X = df.drop(columns=["cantidad"])
    y = df["cantidad"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Cargar hiperparámetros óptimos
    with open(MODEL_PARAMS_PATH, "r") as f:
        best_params = json.load(f)

    # Convertir a DMatrix
    dtrain = DMatrix(X_train, label=y_train)
    dtest = DMatrix(X_test, label=y_test)

    # Preparar parámetros XGBoost según ML_DEVICE (sin heurísticas)
    if ml_device == "gpu":
        params = {
            "tree_method": "gpu_hist",
            "device": "cuda",
            "predictor": "gpu_predictor",
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": float(best_params["learning_rate"]),
            "max_depth": int(best_params["max_depth"]),
            "subsample": float(best_params["subsample"]),
            "colsample_bytree": float(best_params["colsample_bytree"]),
            "gamma": float(best_params["gamma"]),
            "min_child_weight": int(best_params["min_child_weight"]),
        }
    else:
        params = {
            "tree_method": "hist",
            "device": "cpu",
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": float(best_params["learning_rate"]),
            "max_depth": int(best_params["max_depth"]),
            "subsample": float(best_params["subsample"]),
            "colsample_bytree": float(best_params["colsample_bytree"]),
            "gamma": float(best_params["gamma"]),
            "min_child_weight": int(best_params["min_child_weight"]),
        }

    num_boost_round = int(best_params["n_estimators"])
    evals_result = {}

    print("Entrenando modelo XGBoost optimizado…")

    # ------------------------------
    #  MLflow RUN
    # ------------------------------
    with mlflow.start_run():

        # Log params
        mlflow.log_params(best_params)

        # Entrenar
        model = train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dtest, "test")],
            early_stopping_rounds=30,
            verbose_eval=50,
            evals_result=evals_result,
        )

        # RMSE final
        final_rmse = evals_result["test"]["rmse"][-1]
        mlflow.log_metric("rmse", final_rmse)
        print(f"RMSE final test: {final_rmse:.4f}")

        # Guardamos curva de entrenamiento
        joblib.dump(evals_result, EVALS_PATH)

        # ------------------------------
        #  SIGNATURE & INPUT EXAMPLE
        # ------------------------------
        # MLflow no puede inferir desde DMatrix → usar pandas
        signature = infer_signature(X_train, model.predict(dtrain))
        input_example = X_train.iloc[:5]

        # ------------------------------
        #  REGISTRAR MODELO EN MLFLOW
        # ------------------------------
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="xgb_cantidad",
        )

        print("Modelo registrado correctamente en MLflow Registry.")

        # Guardado local (opcional)
        model.save_model(MODEL_PATH)

    print("Entrenamiento completado.")


# ------------------------------
#  MAIN
# ------------------------------
if __name__ == "__main__":
    model_train()
