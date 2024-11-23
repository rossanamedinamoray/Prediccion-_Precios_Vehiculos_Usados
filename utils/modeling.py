from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    linear_model = LinearRegression()
    random_forest_model = RandomForestRegressor(random_state=42, n_estimators=100)

    # Entrenaramos el modelo
    linear_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    # Hacemos las predicciones
    linear_predictions = linear_model.predict(X_test)
    random_forest_predictions = random_forest_model.predict(X_test)

    # Mostramos los resultados
    metrics = {
        "Linear Regression": {
            "MSE": mean_squared_error(y_test, linear_predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, linear_predictions)),
            "R^2": r2_score(y_test, linear_predictions)
        },
        "Random Forest": {
            "MSE": mean_squared_error(y_test, random_forest_predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, random_forest_predictions)),
            "R^2": r2_score(y_test, random_forest_predictions)
        }
    }
    return metrics, random_forest_model

def optimize_model(model, X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               scoring='neg_mean_squared_error',
                               cv=3,
                               n_jobs=-1,
                               verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_