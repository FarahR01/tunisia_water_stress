"""Model evaluation utilities for regression models.

Provides functions for computing metrics, plotting results, and saving evaluation outputs.
"""

import os
from typing import Any, Dict, List

import matplotlib

# Use a non-interactive backend for scripts/CI (prevents GUI windows)
try:
    matplotlib.use("Agg")
except Exception:
    pass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression evaluation metrics.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        Dict[str, float]: Dictionary with MAE, RMSE, and R2 scores.
    """
    mae = mean_absolute_error(y_true, y_pred)
    # use sqrt of MSE for RMSE to be compatible with different sklearn versions
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def plot_actual_vs_pred(
    years: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, out_path: str
) -> None:
    """Plot actual vs predicted values.

    Args:
        years: Array of year values for x-axis.
        y_true: True target values.
        y_pred: Predicted target values.
        out_path: Path to save the plot.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(years, y_true, marker="o", label="Actual")
    plt.plot(years, y_pred, marker="o", label="Predicted")
    plt.xlabel("Year")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_feature_importance(model: Any, feature_names: List[str], out_path: str) -> None:
    """Plot model feature importances as a horizontal bar chart.

    Args:
        model: Trained sklearn model with feature_importances_ attribute.
        feature_names: List of feature names.
        out_path: Path to save the plot.
    """
    if not hasattr(model, "feature_importances_"):
        return
    fi = model.feature_importances_
    order = np.argsort(fi)[::-1]
    names = [feature_names[i] for i in order]
    vals = fi[order]
    plt.figure(figsize=(8, max(3, len(names) * 0.3)))
    plt.barh(names, vals)
    plt.xlabel("Importance")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def save_metrics(metrics: Dict[str, Any], out_path: str) -> None:
    """Save metrics to a CSV file.

    Args:
        metrics: Dictionary of metrics. Can be either:
                 - A single dict of {metric_name: value}
                 - A nested dict of {model_name: {metric_name: value}}
        out_path: Path to save the CSV file.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # If metrics is a mapping of model_name -> {metric: value},
    # convert to a table with one row per model.
    if isinstance(metrics, dict) and metrics and all(isinstance(v, dict) for v in metrics.values()):
        df = pd.DataFrame.from_dict(metrics, orient="index")
        df.index.name = "model"
        df = df.reset_index()
    else:
        # fallback: let pandas try to construct a DataFrame
        df = pd.DataFrame(metrics)
    df.to_csv(out_path, index=False)
