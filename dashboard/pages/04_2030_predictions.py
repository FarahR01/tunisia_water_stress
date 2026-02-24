"""2030 Predictions page - Scenario analysis."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.viz import comparison_table, metric_card, plot_scenario_comparison
from utils.data_loader import (
    load_historical_data,
    load_predictions_2030,
    load_trained_models,
    prepare_features,
)


def run():
    """Run the 2030 predictions page."""
    st.set_page_config(page_title="2030 Predictions", layout="wide")

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

    st.title("🔮 2030 Water Stress Predictions")
    st.markdown(
        "Interactive scenario builder to explore future water stress under different conditions"
    )
    st.markdown("---")

    try:
        df_predictions = load_predictions_2030()
        df_historical = load_historical_data()
        models = load_trained_models()

        st.subheader("📊 Scenario Analysis")

        # Create scenario selector
        col1, col2, col3 = st.columns(3)

        with col1:
            scenario = st.selectbox("Select Scenario", ["Conservative", "Expected", "Pessimistic"])

        with col2:
            target_year = st.slider("Target Year", 2025, 2040, 2030)

        with col3:
            model_selection = st.multiselect(
                "Select Models",
                ["All"]
                + list(
                    df_predictions["model"].unique() if "model" in df_predictions.columns else []
                ),
                default="All",
            )

        st.markdown("---")

        if not df_predictions.empty:
            # Display predictions table
            st.subheader("🎯 Model Predictions")
            comparison_table(df_predictions)

            st.markdown("---")

            # Scenario summary
            st.subheader("📈 Scenario Summary for 2030")

            preds = df_predictions["prediction_2030"]

            col1, col2, col3, col4 = st.columns(4)

            metric_card(col1, "Average", f"{preds.mean():.1f}%")
            metric_card(col2, "Optimistic", f"{preds.min():.1f}%")
            metric_card(col3, "Pessimistic", f"{preds.max():.1f}%")
            metric_card(col4, "Uncertainty", f"±{preds.std():.1f}%")

            st.markdown("---")

            # Key insights
            st.subheader("💡 Key Insights")

            recent_stress = df_historical[df_historical["Year"] >= 2020]
            target_col = "Level of water stress: freshwater withdrawal as a proportion of available freshwater resources"

            if not recent_stress.empty:
                current = recent_stress[target_col].iloc[-1]
                avg_pred = preds.mean()
                change = avg_pred - current

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(
                        f"""
                    **Current Water Stress (2024):** {current:.1f}%

                    **Predicted 2030 Average:** {avg_pred:.1f}%

                    **Expected Change:** {change:+.1f}% ({change/current*100:+.1f}%)
                    """
                    )

                with col2:
                    if change > 0:
                        st.warning(
                            f"⚠️ Water stress is expected to INCREASE by ~{abs(change):.1f}% by 2030"
                        )
                    elif change < 0:
                        st.success(
                            f"✓ Water stress is expected to DECREASE by ~{abs(change):.1f}% by 2030"
                        )
                    else:
                        st.info(f"→ Water stress is expected to REMAIN relatively stable")

    except Exception as e:
        st.error(f"Error loading predictions: {e}")


if __name__ == "__main__":
    run()
