"""Streamlit Dashboard - Main Entry Point

Tunisia Water Stress ML Analysis
Interactive dashboard for water stress prediction and analysis.
"""

import sys
from pathlib import Path

import streamlit as st

# Add dashboard to path
sys.path.insert(0, str(Path(__file__).parent))

from components.viz import comparison_table, metric_card, plot_historical_trend
from utils.data_loader import (
    calculate_statistics,
    get_feature_columns,
    load_historical_data,
    load_predictions_2030,
    load_trained_models,
)

# Configure page
st.set_page_config(
    page_title="Tunisia Water Stress Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://docs.streamlit.io",
        "Report a bug": None,
        "About": """
        **Tunisia Water Stress ML Dashboard**

        Interactive analysis of Tunisia's water stress levels using machine learning.
        Data Source: World Bank (1961-2024) | Models: 5 ML algorithms
        """,
    },
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    /* Main colors */
    :root {
        --primary: #0F766E;
        --teal: #06B6D4;
        --green: #10B981;
        --accent: #F59E0B;
    }

    /* Header styling */
    h1 {
        color: #0F766E;
        font-weight: 800;
        letter-spacing: -1px;
    }

    h2 {
        color: #0F766E;
        font-weight: 700;
        border-bottom: 3px solid #06B6D4;
        padding-bottom: 0.5rem;
    }

    h3 {
        color: #0F766E;
    }

    /* Metric card styling */
    [data-testid="metric-container"] {
        background-color: #F0FDFA;
        border-left: 4px solid #0F766E;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(15, 118, 110, 0.1);
    }

    /* Expander styling */
    [data-testid="expanderContent"] {
        padding: 1rem;
    }

    /* Main background */
    .main {
        background-color: #FFFFFF;
    }

    /* Divider */
    hr {
        border: 1px solid #E0F2F1;
    }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    """Main application entry point."""

    # Sidebar
    with st.sidebar:
        st.title("🗺️ Navigation")
        st.markdown(
            """
        Use the page menu above to navigate between sections:

        - **Home**: Overview and key metrics
        - **Historical Analysis**: Data exploration
        - **Model Comparison**: Performance metrics
        - **2030 Predictions**: Scenario analysis
        - **Feature Analysis**: Interpretability
        """
        )

        st.divider()

        st.subheader("ℹ️ About This Dashboard")
        st.markdown(
            """
        **Purpose:** Analyze Tunisia's water stress and predict 2030 scenarios

        **Data Period:** 1961-2024 (60+ years)

        **Models:** 5 ML algorithms with ensemble predictions

        **Target:** Water stress levels (freshwater withdrawal as a % of available)
        """
        )

        st.divider()

        st.subheader("📊 Dashboard Stats")

        try:
            df = load_historical_data()
            models = load_trained_models()
            features = get_feature_columns(df)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Data Points", len(df))
            with col2:
                st.metric("Features", len(features))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Models", len(models))
            with col2:
                st.metric("Time Span", "60+ years")

        except Exception as e:
            st.warning(f"Could not load stats: {e}")

    # Main content
    st.title("💧 Tunisia Water Stress Analysis Dashboard")
    st.markdown("**Interactive machine learning insights for water resource management**")
    st.markdown("---")

    try:
        df_historical = load_historical_data()
        models = load_trained_models()
        df_predictions = load_predictions_2030()
        features = get_feature_columns(df_historical)

        # Key metrics row
        st.subheader("📊 Quick Overview")
        col1, col2, col3, col4, col5 = st.columns(5)

        target_col = "Level of water stress: freshwater withdrawal as a proportion of available freshwater resources"
        stats = calculate_statistics(df_historical, target_col)

        with col1:
            st.metric("Current Stress", f"{stats['latest']:.1f}%")
        with col2:
            st.metric("Historical Avg", f"{stats['mean']:.1f}%")
        with col3:
            st.metric("Data Points", len(df_historical))
        with col4:
            st.metric("Models", len(models))
        with col5:
            st.metric("Features", len(features))

        st.markdown("---")

        # Feature highlights
        st.subheader("✨ Dashboard Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
            ### 📈 Historical Analysis
            Explore 60+ years of water stress data with interactive visualizations and trend analysis.
            """
            )

        with col2:
            st.markdown(
                """
            ### 🤖 Model Comparison
            Compare predictions from 5 ML models with detailed performance metrics and insights.
            """
            )

        with col3:
            st.markdown(
                """
            ### 🔮 2030 Predictions
            Interactive scenario builder to explore future water stress under different conditions.
            """
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
            ### 🎯 Feature Analysis
            Understand which factors drive water stress with feature importance and SHAP analysis.
            """
            )

        with col2:
            st.markdown(
                """
            ### 📊 Ensemble Predictions
            Robust predictions combining multiple models with confidence intervals.
            """
            )

        with col3:
            st.markdown(
                """
            ### 🚀 Production Ready
            REST API, Docker deployment, and comprehensive documentation included.
            """
            )

        st.markdown("---")

        # Historical trend chart
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

        # Quick start section
        st.subheader("🚀 Quick Start")

        st.markdown(
            """
        **Select a page from the sidebar to get started:**

        1. **Home** - Return to this overview
        2. **Historical Analysis** - Explore historical water stress patterns (1961-2024)
        3. **Model Comparison** - Compare ML model performance
        4. **2030 Predictions** - Build custom scenarios and see predictions
        5. **Feature Analysis** - Understand feature importance and relationships

        **Key Insight:** The dashboard uses a time-aware train/test split (1960-2010 training, 2011-2024 testing)
        to prevent data leakage and provide realistic performance estimates.
        """
        )

        st.markdown("---")

        # About section
        with st.expander("ℹ️ About This Analysis"):
            st.markdown(
                """
            ### Problem Statement
            Tunisia faces significant water stress due to growing population, climate change, and agricultural demands.
            This project builds ML models to **predict future water stress levels** and help policymakers plan interventions.

            ### Data
            - **Source:** World Bank Open Data
            - **Period:** 1961-2024 (60+ years)
            - **Frequency:** Annual
            - **Indicator:** Freshwater withdrawal as % of available freshwater resources

            ### Models Used
            | Model | Type | Best For |
            |-------|------|----------|
            | Random Forest | Ensemble | Production predictions |
            | Decision Tree | Tree-based | Interpretability |
            | Linear Regression | Linear | Baseline comparison |
            | Ridge | Regularized | Multicollinearity handling |
            | Lasso | Regularized | Feature selection |

            ### Key Features
            - ✅ Time-aware train/test split (prevents data leakage)
            - ✅ Target leakage detection
            - ✅ Multicollinearity analysis
            - ✅ Hyperparameter tuning (GridSearchCV)
            - ✅ Feature importance & partial dependence
            - ✅ Ensemble predictions with confidence intervals
            - ✅ Interactive visualizations
            - ✅ REST API & Docker deployment
            """
            )

        st.markdown("---")

        # Developer info
        st.subheader("👩‍💼 Built By")
        st.markdown(
            """
        **Farah Rihane** | Software Engineer discovering AI/ML/Data Science

        This project demonstrates a complete ML lifecycle from data exploration to production-ready dashboards.
        """
        )

    except Exception as e:
        st.error(f"Error loading dashboard: {e}")
        st.info("Please ensure all data files and models are properly loaded.")


if __name__ == "__main__":
    main()
