"""Factory for creating ML models with optional hyperparameter tuning.

Centralizes model instantiation, configuration, and GridSearchCV setup.
"""
from __future__ import annotations
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from config import ML_HYPERPARAMETER_GRIDS, ML_PARAMS


def create_ml_models(tune_hyperparameters=False, saved_best_params=None):
    """Create a dictionary of ML models for strategy testing.
    
    Args:
        tune_hyperparameters: If True, returns GridSearchCV wrapped models for tuning.
                             If False, returns models with default parameters.
        saved_best_params: Dict mapping model names to their best parameters from previous runs.
                          If provided, these params will be used instead of tuning.
    
    Returns a dict with model_name -> (model_instance, short_name) pairs.
    """
    if saved_best_params is None:
        saved_best_params = {}
    
    random_seed = ML_PARAMS.get('random_seed', 42)
    
    base_models = {
        'Linear Regression': (LinearRegression(), 'LinReg'),
        'Elastic Net': (ElasticNet(random_state=random_seed), 'ElasticNet'),
        'LASSO': (Lasso(random_state=random_seed), 'LASSO'),
        'XGBoost' : (XGBRegressor(random_state=random_seed, n_jobs=ML_PARAMS.get('n_jobs', -1)), 'XGB'),
        # 'Support Vector Machine': (SVR(), 'SVM'),
        'K-Nearest Neighbor': (KNeighborsRegressor(), 'KNN'),
        'Decision Tree': (DecisionTreeRegressor(random_state=random_seed), 'DTree'),
        'Extra Trees': (ExtraTreesRegressor(random_state=random_seed), 'ExtraTrees'),
        'Random Forest': (RandomForestRegressor(random_state=random_seed), 'RF'),
        'Gradient Boosting': (GradientBoostingRegressor(random_state=random_seed), 'GBT'),
        'Adaptive Boosting': (AdaBoostRegressor(random_state=random_seed), 'AdaBoost'),
    }
    
    tuned_models = {}
    
    for model_name, (model, short_name) in base_models.items():
        # Check if we have saved best params for this model
        if model_name in saved_best_params:
            print(f"  → Using saved best params for {model_name}: {saved_best_params[model_name]}")
            model.set_params(**saved_best_params[model_name])
            tuned_models[model_name] = (model, short_name)
        elif tune_hyperparameters:
            # Only tune if we don't have saved params
            param_grid = ML_HYPERPARAMETER_GRIDS.get(model_name, {})
            
            if param_grid:
                # Use GridSearchCV with 3-fold CV, optimize for negative MSE
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    scoring='neg_mean_squared_error',
                    n_jobs=ML_PARAMS.get('n_jobs', -1),
                    verbose=0
                )
                tuned_models[model_name] = (grid_search, short_name)
            else:
                # No hyperparameters to tune (e.g., LinearRegression)
                tuned_models[model_name] = (model, short_name)
        else:
            # No tuning, just use default params
            tuned_models[model_name] = (model, short_name)
    
    return tuned_models


__all__ = ["create_ml_models"]
