"""Home page - Overview and key metrics."""

import sys
from pathlib import Path

import streamlit as st

# Add dashboard to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.viz import comparison_table, metric_card, plot_historical_trend
from utils.data_loader import (
    calculate_statistics,
    load_historical_data,
    load_predictions_2030,
)


def run():
    """Run the home page."""
    st.set_page_config(page_title="Home", layout="wide")

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

    st.title("💧 Tunisia Water Stress Analysis Dashboard")
    st.markdown("**Real-time ML insights for water resource management**")
    st.markdown("---")

    try:
        # Load data
        df_historical = load_historical_data()
        df_predictions = load_predictions_2030()

        if df_historical.empty:
            st.error("No historical data available")
            return

        # Key statistics
        st.subheader("📊 Key Statistics")

        col1, col2, col3, col4 = st.columns(4)

        target_col = "Level of water stress: freshwater withdrawal as a proportion of available freshwater resources"
        stats = calculate_statistics(df_historical, target_col)

        metric_card(col1, "Current Water Stress", f"{stats['latest']:.1f}%")
        metric_card(col2, "Average (Historical)", f"{stats['mean']:.1f}%")
        metric_card(col3, "Data Range", f"{int(stats['min']):.0f}%-{int(stats['max']):.0f}%")
        metric_card(col4, "Years of Data", len(df_historical))

        st.markdown("---")

        # Historical trend
        st.subheader("📈 Historical Water Stress Trend")
        fig = plot_historical_trend(
            df_historical,
            target_col,
            title="Water Stress Levels (1961-2024)",
            show_ma=True,
            ma_window=5,
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("---")

        # 2030 Predictions
        st.subheader("🔮 2030 Predictions Summary")

        if not df_predictions.empty and "prediction_2030" in df_predictions.columns:
            col1, col2, col3 = st.columns(3)

            avg_prediction = df_predictions["prediction_2030"].mean()
            min_prediction = df_predictions["prediction_2030"].min()
            max_prediction = df_predictions["prediction_2030"].max()

            metric_card(col1, "Average Prediction", f"{avg_prediction:.1f}%")
            metric_card(col2, "Min (Best Case)", f"{min_prediction:.1f}%")
            metric_card(col3, "Max (Worst Case)", f"{max_prediction:.1f}%")

            st.markdown("---")

            st.subheader("Model Predictions")
            comparison_table(df_predictions, title=None)

        st.markdown("---")

        # Overview text
        st.subheader("📋 About This Dashboard")

        with st.expander("View Overview"):
            st.markdown(
                """
            This dashboard provides comprehensive analysis of Tunisia's water stress levels and
            machine learning predictions for future scenarios.

            **Features:**
            - **Historical Analysis**: Explore 60+ years of water stress data
            - **Model Comparison**: Compare predictions from multiple ML models
            - **2030 Scenarios**: Interactive scenario analysis with future projections
            - **Feature Analysis**: Understand which factors drive water stress
            - **Advanced Analytics**: Feature importance and partial dependence analysis

            **Navigation:**
            Use the sidebar menu to explore different sections:
            - 🏠 Home: Overview and key metrics
            - 📊 Historical Analysis: Detailed historical data exploration
            - 🤖 Model Comparison: Model performance and metrics
            - 🔮 2030 Predictions: Future scenarios and projections
            - 🎯 Feature Analysis: Feature importance and interpretability
            """
            )

    except Exception as e:
        st.error(f"Error loading home page: {e}")


if __name__ == "__main__":
    run()
