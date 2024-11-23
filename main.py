import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from utils.preprocessing import load_and_clean_data, preprocess_data
from utils.modeling import train_and_evaluate_models, optimize_model

def main():
    data_path = "data/vehicles-2.csv"

    # Cargamos y limpiamos los datos
    print("Cargando y limpiando datos...")
    dataset = load_and_clean_data(data_path)

    # Preprocesamiento de datos
    print("Preprocesando datos...")
    X, y = preprocess_data(dataset)

    # Dividimos los datos en entrenamiento y prueba
    print("Dividiendo los datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamos los modelos y los evaluamos
    print("Entrenando y evaluando modelos...")
    metrics, random_forest_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Mostrar métricas iniciales
    print("\nMétricas de los modelos:")
    for model, model_metrics in metrics.items():
        print(f"{model}:")
        for metric, value in model_metrics.items():
            print(f"  {metric}: {value:.2f}")

    # Optimizar el modelo Random Forest
    print("\nOptimizando el modelo Random Forest...")
    best_model, best_params = optimize_model(random_forest_model, X_train, y_train)

    # Evaluamos el mejor modelo
    print("\nEvaluando el mejor modelo...")
    best_model_predictions = best_model.predict(X_test)
    best_model_mse = mean_squared_error(y_test, best_model_predictions)
    best_model_rmse = np.sqrt(best_model_mse)
    best_model_r2 = r2_score(y_test, best_model_predictions)

    # Mostrar métricas del modelo optimizado
    print("\nMétricas del modelo optimizado:")
    print(f"Mejores parámetros: {best_params}")
    print(f"MSE: {best_model_mse:.2f}")
    print(f"RMSE: {best_model_rmse:.2f}")
    print(f"R^2: {best_model_r2:.2f}")

if __name__ == "__main__":
    main()