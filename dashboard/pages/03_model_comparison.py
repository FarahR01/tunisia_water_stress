"""Model comparison page - Performance metrics."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.viz import comparison_table, metric_card, plot_model_comparison
from utils.data_loader import (
    load_model_metrics,
    load_predictions_2030,
    load_trained_models,
)


def run():
    """Run the model comparison page."""
    st.set_page_config(page_title="Model Comparison", layout="wide")

    # Custom CSS
    st.markdown(
        """
    <style>
        h2 {
            color: #0F766E;
            border-bottom: 3px solid #06B6D4;
            padding-bottom: 0.5rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("🤖 Model Comparison & Performance Metrics")
    st.markdown("Compare ML model predictions and performance across multiple evaluation metrics")
    st.markdown("---")

    try:
        # Load data
        df_metrics = load_model_metrics()
        df_predictions = load_predictions_2030()

        # Model performance overview
        st.subheader("📊 Model Performance Metrics")

        if not df_metrics.empty:
            comparison_table(df_metrics, title="Detailed Metrics")

            st.markdown("---")

            # Key metrics comparison
            st.subheader("🎯 Key Performance Indicators")

            metrics_to_show = ["MAE", "RMSE", "R2", "MAPE"]
            available_metrics = [m for m in metrics_to_show if m in df_metrics.columns]

            if available_metrics:
                cols = st.columns(min(4, len(available_metrics)))

                for idx, metric_name in enumerate(available_metrics):
                    if not df_metrics[metric_name].isnull().all():
                        best_value = (
                            df_metrics[metric_name].min()
                            if metric_name in ["MAE", "RMSE", "MAPE"]
                            else df_metrics[metric_name].max()
                        )
                        worst_value = (
                            df_metrics[metric_name].max()
                            if metric_name in ["MAE", "RMSE", "MAPE"]
                            else df_metrics[metric_name].min()
                        )

                        with cols[idx % len(cols)]:
                            st.metric(
                                f"Best {metric_name}",
                                f"{best_value:.4f}",
                                delta=f"Worst: {worst_value:.4f}",
                            )

        st.markdown("---")

        # 2030 Predictions comparison
        st.subheader("🔮 2030 Predictions by Model")

        if not df_predictions.empty:
            fig = plot_model_comparison(df_predictions)
            if fig:
                st.plotly_chart(fig, width="stretch")

            st.markdown("---")

            # Prediction statistics
            st.subheader("📈 Prediction Statistics")

            col1, col2, col3, col4 = st.columns(4)

            preds = df_predictions["prediction_2030"]
            metric_card(col1, "Mean Prediction", f"{preds.mean():.1f}%")
            metric_card(col2, "Median Prediction", f"{preds.median():.1f}%")
            metric_card(col3, "Min Prediction", f"{preds.min():.1f}%")
            metric_card(col4, "Max Prediction", f"{preds.max():.1f}%")

            st.markdown("---")

            # Predictions table
            st.subheader("Model Predictions")
            st.dataframe(
                df_predictions.sort_values("prediction_2030", ascending=False),
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Error loading model comparison: {e}")


if __name__ == "__main__":
    run()
