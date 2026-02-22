"""Hyperparameter tuning utilities."""
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    XGBRegressor = None


# Define hyperparameter grids for each model
PARAM_GRIDS = {
    "LinearRegression": {},  # No hyperparameters
    "Ridge": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "Lasso": {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
    },
    "DecisionTree": {
        "max_depth": [3, 5, 7, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "XGBoost": {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
    },
}


def tune_hyperparameters(X_train, y_train, X_test, y_test, model_name, model, cv=5, n_iter=None):
    """
    Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
    
    Args:
        X_train, y_train: training data
        X_test, y_test: test data (used for final evaluation)
        model_name: name of the model
        model: sklearn model instance
        cv: number of cross-validation folds
        n_iter: if provided, use RandomizedSearchCV with n_iter searches
    
    Returns:
        dict with best_model, best_params, cv_results_df
    """
    params = PARAM_GRIDS.get(model_name, {})
    
    if not params:
        # No hyperparameter grid defined — just fit the model normally
        model.fit(X_train, y_train)
        return {
            "best_model": model,
            "best_params": {},
            "best_score": None,
            "cv_results": pd.DataFrame(),
        }
    
    if n_iter and n_iter < len(params) ** 2:
        # Use RandomizedSearchCV for large spaces
        searcher = RandomizedSearchCV(
            model,
            params,
            n_iter=n_iter,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            verbose=1,
            random_state=42,
            refit=True,
        )
    else:
        # Use GridSearchCV
        searcher = GridSearchCV(
            model,
            params,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
    
    print(f"Tuning hyperparameters for {model_name}...")
    searcher.fit(X_train, y_train)
    
    best_model = searcher.best_estimator_
    cv_results_df = pd.DataFrame(searcher.cv_results_)
    
    return {
        "best_model": best_model,
        "best_params": searcher.best_params_,
        "best_score": searcher.best_score_,
        "cv_results": cv_results_df,
    }


def save_hyperparameter_results(tuning_results, out_dir):
    """Save hyperparameter tuning results to CSV."""
    os.makedirs(out_dir, exist_ok=True)
    
    summary = []
    for model_name, result in tuning_results.items():
        if result["best_params"]:
            summary.append({
                "model": model_name,
                "best_score": result["best_score"],
                "hyperparameters": str(result["best_params"]),
            })
    
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(out_dir, "hyperparameter_tuning_summary.csv"), index=False)
        
        # Also save detailed CV results for each model
        for model_name, result in tuning_results.items():
            if not result["cv_results"].empty:
                result["cv_results"].to_csv(
                    os.path.join(out_dir, f"{model_name}_cv_results.csv"),
                    index=False,
                )
