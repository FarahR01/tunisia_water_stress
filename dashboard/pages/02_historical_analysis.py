"""Historical analysis page - Data exploration."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.viz import plot_correlation_heatmap, plot_historical_trend
from utils.data_loader import get_feature_columns, load_historical_data


def run():
    """Run the historical analysis page."""
    st.set_page_config(page_title="Historical Analysis", layout="wide")

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

    st.title("📈 Historical Water Stress Analysis")
    st.markdown(
        "Explore 60+ years of water stress data with interactive visualizations and trend analysis"
    )
    st.markdown("---")

    try:
        df = load_historical_data()

        if df.empty:
            st.error("No data available")
            return

        target_col = "Level of water stress: freshwater withdrawal as a proportion of available freshwater resources"

        # Basic statistics
        st.subheader("📊 Data Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Data Points", len(df))
        col2.metric("Year Range", f"{int(df['Year'].min())}-{int(df['Year'].max())}")
        col3.metric("Mean Stress", f"{df[target_col].mean():.1f}%")
        col4.metric("Std Dev", f"{df[target_col].std():.1f}%")

        st.markdown("---")

        # Historical trend
        st.subheader("Historical Trend with Moving Average")

        ma_window = st.slider("Moving Average Window", 3, 10, 5)
        fig = plot_historical_trend(df, target_col, show_ma=True, ma_window=ma_window)
        st.plotly_chart(fig, width="stretch")

        st.markdown("---")

        # Detailed statistics
        st.subheader("📉 Detailed Statistics by Period")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**1961-1990 (Early Period)**")
            early = df[df["Year"] <= 1990]
            if not early.empty:
                st.write(f"Mean: {early[target_col].mean():.1f}%")
                st.write(f"Min: {early[target_col].min():.1f}%")
                st.write(f"Max: {early[target_col].max():.1f}%")

        with col2:
            st.markdown("**2010-2024 (Recent Period)**")
            recent = df[df["Year"] >= 2010]
            if not recent.empty:
                st.write(f"Mean: {recent[target_col].mean():.1f}%")
                st.write(f"Min: {recent[target_col].min():.1f}%")
                st.write(f"Max: {recent[target_col].max():.1f}%")

        st.markdown("---")

        # Correlation analysis
        st.subheader("🔗 Feature Relationships")

        fig_corr = plot_correlation_heatmap(df, title="Feature Correlation Matrix")
        if fig_corr:
            st.plotly_chart(fig_corr, width="stretch")

        st.markdown("---")

        # Data table
        st.subheader("📋 Raw Data")

        st.dataframe(df.sort_values("Year", ascending=False), use_container_width=True)

    except Exception as e:
        st.error(f"Error loading historical analysis: {e}")


if __name__ == "__main__":
    run()
