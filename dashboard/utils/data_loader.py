"""Data loading utilities with caching for Streamlit dashboard."""

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_project_paths():
    """Get paths to key project directories."""
    return {
        "data": PROJECT_ROOT / "data" / "cleaned_water_stress.csv",
        "models_dir": PROJECT_ROOT / "models_tuned",
        "predictions": PROJECT_ROOT
        / "artifacts"
        / "predictions"
        / "water_stress_2030_predictions.csv",
        "feature_importance": PROJECT_ROOT
        / "artifacts"
        / "models_tuned"
        / "feature_importance_summary.csv",
        "metrics": PROJECT_ROOT / "artifacts" / "models_tuned" / "metrics.csv",
    }


def load_historical_data() -> pd.DataFrame:
    """Load historical water stress data."""
    paths = get_project_paths()
    df = pd.read_csv(paths["data"])
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df.sort_values("Year").reset_index(drop=True)


def load_predictions_2030() -> pd.DataFrame:
    """Load 2030 predictions."""
    paths = get_project_paths()
    try:
        df = pd.read_csv(paths["predictions"])
        return df
    except FileNotFoundError:
        logger.warning(f"Predictions file not found: {paths['predictions']}")
        return pd.DataFrame()


def load_feature_importance() -> pd.DataFrame:
    """Load feature importance data."""
    paths = get_project_paths()
    try:
        df = pd.read_csv(paths["feature_importance"])
        return df
    except FileNotFoundError:
        logger.warning(f"Feature importance file not found: {paths['feature_importance']}")
        return pd.DataFrame()


def load_model_metrics() -> pd.DataFrame:
    """Load model evaluation metrics."""
    paths = get_project_paths()
    try:
        df = pd.read_csv(paths["metrics"])
        return df
    except FileNotFoundError:
        logger.warning(f"Metrics file not found: {paths['metrics']}")
        return pd.DataFrame()


def load_trained_models() -> Dict:
    """Load all trained models."""
    paths = get_project_paths()
    models_dir = paths["models_dir"]

    models = {}
    model_files = {
        "DecisionTree": "DecisionTree.joblib",
        "Lasso": "Lasso.joblib",
        "LinearRegression": "LinearRegression.joblib",
        "RandomForest": "RandomForest.joblib",
        "Ridge": "Ridge.joblib",
    }

    for model_name, filename in model_files.items():
        model_path = models_dir / filename
        if model_path.exists():
            try:
                models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
        else:
            logger.warning(f"Model file not found: {model_path}")

    return models


def calculate_statistics(df: pd.DataFrame, column: str) -> Dict:
    """Calculate statistics for a column."""
    return {
        "mean": df[column].mean(),
        "median": df[column].median(),
        "std": df[column].std(),
        "min": df[column].min(),
        "max": df[column].max(),
        "latest": df[column].iloc[-1] if len(df) > 0 else None,
    }


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature column names (exclude Year and target, then select first 5 to match model training)."""
    exclude_cols = {
        "Year",
        "Level of water stress: freshwater withdrawal as a proportion of available freshwater resources",
    }

    all_features = [col for col in df.columns if col not in exclude_cols]
    return all_features[:5]


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, list]:
    """Prepare feature matrix and column names. Returns first 5 features to match model training."""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(df[feature_cols].mean()).values
    return X, feature_cols
