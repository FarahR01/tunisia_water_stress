"""Model utilities for predictions and analysis."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_feature_importance(
    model, feature_names: List[str], model_name: str = ""
) -> pd.DataFrame:
    """Extract feature importance from a trained model."""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            logger.warning(f"Model {model_name} does not have feature importance")
            return pd.DataFrame()

        # Normalize to 0-1 range
        importances = (
            importances / np.max(np.abs(importances))
            if np.max(np.abs(importances)) > 0
            else importances
        )

        df = pd.DataFrame(
            {
                "feature": feature_names[: len(importances)],
                "importance": importances[: len(feature_names)],
                "model": model_name,
            }
        ).sort_values("importance", ascending=False)

        return df
    except Exception as e:
        logger.error(f"Error extracting importance from {model_name}: {e}")
        return pd.DataFrame()


def get_partial_dependence_curve(
    model, X: np.ndarray, feature_idx: int, feature_name: str, n_points: int = 30
) -> pd.DataFrame:
    """Generate partial dependence curve for a feature."""
    try:
        feature_values = X[:, feature_idx]
        unique_vals = np.linspace(feature_values.min(), feature_values.max(), n_points)

        predictions = []
        for val in unique_vals:
            X_modified = X.copy()
            X_modified[:, feature_idx] = val
            try:
                pred = model.predict(X_modified).mean()
            except:
                pred = np.nan
            predictions.append(pred)

        return pd.DataFrame(
            {"feature_value": unique_vals, "prediction": predictions, "feature": feature_name}
        )
    except Exception as e:
        logger.error(f"Error computing partial dependence for {feature_name}: {e}")
        return pd.DataFrame()


def calculate_ensemble_prediction(
    models: Dict, X: np.ndarray
) -> Tuple[float, float, Tuple[float, float]]:
    """Calculate ensemble prediction from multiple models."""
    predictions = []

    for model_name, model in models.items():
        try:
            # Handle feature names warning by using DataFrame view
            if hasattr(model, "feature_names_in_"):
                pred = np.mean(model.predict(X))
            else:
                pred = np.mean(model.predict(X))
            predictions.append(pred)
        except Exception as e:
            logger.warning(f"Could not predict with {model_name}: {e}")

    if not predictions:
        return np.nan, np.nan, (np.nan, np.nan)

    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    ci_lower = mean_pred - 1.96 * std_pred
    ci_upper = mean_pred + 1.96 * std_pred

    return mean_pred, std_pred, (ci_lower, ci_upper)


def calculate_prediction_bounds(
    predictions: List[float], confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate confidence interval for predictions."""
    predictions = np.array(predictions)
    mean = np.mean(predictions)
    std = np.std(predictions)

    # Approximate z-score for 95% confidence (1.96)
    margin = 1.96 * std / np.sqrt(len(predictions))

    return mean - margin, mean + margin
