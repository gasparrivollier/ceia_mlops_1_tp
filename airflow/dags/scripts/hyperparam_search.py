import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import randint, uniform

DATA_PATH = "/opt/airflow/dags/data/processed.parquet"
MODEL_PARAMS_PATH = "/opt/airflow/dags/models/best_model_params.json"

def hyperparam_search():

    # La variable de entorno ML_DEVICE debe indicar 'gpu' o 'cpu'.
    ml_device = os.environ.get("ML_DEVICE")
    if ml_device is None:
        raise RuntimeError("La variable de entorno ML_DEVICE no está definida. Configura ML_DEVICE=gpu o ML_DEVICE=cpu en el archivo .env")
    ml_device = ml_device.strip().lower()
    if ml_device not in ("gpu", "cpu"):
        raise RuntimeError("Valor inválido para ML_DEVICE. Usa 'gpu' o 'cpu'.")

    df = pd.read_parquet(DATA_PATH)

    # 2️⃣ Definir variables
    X = df.loc[:, df.columns != 'cantidad']
    y = df['cantidad']

    # 3️⃣ Train-test split + escalado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4️⃣ Definir el estimador según ML_DEVICE (sin heurísticas)
    if ml_device == "gpu":
        xgb_est = XGBRegressor(
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            device='cuda',
            n_jobs=1,
            eval_metric='rmse',
            verbosity=1
        )
    else:
        xgb_est = XGBRegressor(
            tree_method='hist',
            device='cpu',
            n_jobs=max(1, (os.cpu_count() or 1) - 1),
            eval_metric='rmse',
            verbosity=1
        )

    # 5️⃣ Espacio de búsqueda
    param_dist = {
        'n_estimators': randint(400, 1200),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(4, 15),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 1),
        'min_child_weight': randint(1, 10)
    }

    # 6️⃣ Búsqueda aleatoria
    # Elegir n_jobs para RandomizedSearchCV (si GPU: n_jobs=1)
    # Para evitar problemas con multiprocessing en contenedores limitados,
    # ejecutamos secuencialmente en modo debug: n_jobs=1 y menos iteraciones.
    random_search = RandomizedSearchCV(
        estimator=xgb_est,
        param_distributions=param_dist,
        n_iter=8,
        scoring='r2',
        cv=3,
        verbose=2,
        n_jobs=1,
        random_state=42
    )

    print(f"Ejecutando búsqueda aleatoria en modo secuencial de debug (ML_DEVICE={ml_device}, n_iter=8)...")
    random_search.fit(X_train_scaled, y_train)

    # 7️⃣ Guardar hiperparámetros
    best_params = random_search.best_params_

    with open(MODEL_PARAMS_PATH, 'w') as f:
        json.dump(best_params, f)

    print(f"✔️ Mejores hiperparámetros guardados en {MODEL_PARAMS_PATH}")

    return best_params

if __name__ == "__main__":
    hyperparam_search()