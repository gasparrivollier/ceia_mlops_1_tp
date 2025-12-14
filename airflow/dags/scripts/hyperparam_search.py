import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
import mlflow
import mlflow.xgboost

DATA_PATH = "/opt/airflow/dags/data/processed.parquet"
MODEL_PARAMS_PATH = "/opt/airflow/dags/models/best_model_params.json"
MLFLOW_TRACKING_URI = "http://mlflow:5000" 
MLFLOW_EXPERIMENT_NAME = "XGBoost_Param_Tuning"

def hyperparam_search():

    # Defino trackeo en MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Habilito logging automático en MLflow.
    # log_models=False para guardar resultados de cada corrida 
    mlflow.sklearn.autolog(log_models=False, log_input_examples=True)

    # Cargar dataset
    df = pd.read_parquet(DATA_PATH)

    # Definir variables 
    X = df.loc[:, df.columns != 'cantidad']
    y = df['cantidad'].to_numpy()

    # Train-test split + escalado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Definir el modelo sin GPU
    xgb_model = XGBRegressor(
        tree_method='hist',
        device='cpu',
        n_jobs=-1,
        eval_metric='rmse',
        verbosity=1
    )

    # Espacio de búsqueda
    param_dist = {
        'n_estimators': randint(400, 1200),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(4, 15),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 1),
        'min_child_weight': randint(1, 10)
    }

    # Búsqueda aleatoria
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=30,
        scoring='r2',
        cv=3,
        verbose=2,
        n_jobs=1
    )

    print("Ejecutando búsqueda de hyperparametros...")

    # Run padre en MLflow

    with mlflow.start_run(run_name="xgb_hyperparam_search") as parent_run:

        random_search.fit(X_train_scaled, y_train)

        # Extraer hiperparámetros y score
        best_params = random_search.best_params_
        best_score = random_search.best_score_

        # Loguear los mejores hiperparámetros y score
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_r2", best_score)
        
         # Guardar hiperparámetros en JSON
        with open(MODEL_PARAMS_PATH, 'w') as f:
            json.dump(best_params, f)

        print(f"✔️ Mejores hiperparámetros guardados en {MODEL_PARAMS_PATH}")

    return best_params

if __name__ == "__main__":
    hyperparam_search()