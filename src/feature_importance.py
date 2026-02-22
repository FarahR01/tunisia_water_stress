"""Feature importance analysis and extraction."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def extract_feature_importance(model, feature_names, model_name):
    """
    Extract feature importance from a model.
    Returns a DataFrame with columns: feature, importance, rank
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["model"] = model_name
    return df


def plot_top_features(models_dict, feature_names, X_test, top_n=10, out_dir="models"):
    """
    Plot top N features by importance for each model that supports it.
    Creates individual plots and a comparison plot.
    """
    os.makedirs(out_dir, exist_ok=True)

    importance_dfs = []

    for model_name, model in models_dict.items():
        df = extract_feature_importance(model, feature_names, model_name)
        if df is not None:
            importance_dfs.append(df)

            # Plot top N for this model
            top_df = df.head(top_n)
            plt.figure(figsize=(10, max(5, top_n * 0.3)))
            plt.barh(range(len(top_df)), top_df["importance"].values)
            plt.yticks(range(len(top_df)), top_df["feature"].values)
            plt.xlabel("Importance")
            plt.title(f"Top {top_n} Feature Importances - {model_name}")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{model_name}_top_features.png"), dpi=100)
            plt.close()

    # Comparison plot if multiple models available
    if len(importance_dfs) > 1:
        comparison_df = pd.concat(importance_dfs, ignore_index=True)

        # Get top N averaged features
        top_features = (
            comparison_df.groupby("feature")["importance"].mean().nlargest(top_n).index.tolist()
        )

        filtered = comparison_df[comparison_df["feature"].isin(top_features)]

        plt.figure(figsize=(12, max(6, top_n * 0.35)))
        for model_name in filtered["model"].unique():
            model_data = filtered[filtered["model"] == model_name].sort_values(
                "importance", ascending=True
            )
            plt.barh(
                model_data["feature"].values,
                model_data["importance"].values,
                label=model_name,
                alpha=0.7,
            )

        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importances - Model Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "feature_importance_comparison.png"), dpi=100)
        plt.close()

    return importance_dfs


def save_feature_importance_summary(importance_dfs, out_path):
    """Save feature importance as CSV for auditing."""
    if not importance_dfs:
        return

    summary_df = pd.concat(importance_dfs, ignore_index=True)
    summary_df = summary_df.sort_values(["model", "rank"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary_df.to_csv(out_path, index=False)
