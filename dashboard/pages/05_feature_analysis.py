"""Feature analysis page - Interpretability and feature importance."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.viz import comparison_table, plot_feature_importance
from utils.data_loader import (
    get_feature_columns,
    load_feature_importance,
    load_historical_data,
    load_trained_models,
    prepare_features,
)
from utils.model_utils import extract_feature_importance, get_partial_dependence_curve


def run():
    """Run the feature analysis page."""
    st.set_page_config(page_title="Feature Analysis", layout="wide")

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

    st.title("🎯 Feature Analysis & Interpretability")
    st.markdown(
        "Understand which factors drive water stress with feature importance and correlation analysis"
    )
    st.markdown("---")

    try:
        df_importance = load_feature_importance()
        df_historical = load_historical_data()
        models = load_trained_models()

        # Feature importance section
        st.subheader("📊 Feature Importance")

        if not df_importance.empty:
            comparison_table(df_importance, title="Feature Importance Summary")

            st.markdown("---")

            fig = plot_feature_importance(df_importance, top_n=10)
            if fig:
                st.plotly_chart(fig, width="stretch")

        st.markdown("---")

        # Partial dependence section
        st.subheader("📈 Partial Dependence Analysis")

        feature_cols = get_feature_columns(df_historical)

        if feature_cols and models:
            selected_feature = st.selectbox("Select Feature", feature_cols)
            selected_model = st.selectbox("Select Model", list(models.keys()))

            if selected_feature and selected_model in models:
                try:
                    X, features = prepare_features(df_historical)
                    feature_idx = features.index(selected_feature)

                    model = models[selected_model]
                    pd_df = get_partial_dependence_curve(
                        model, X, feature_idx, selected_feature, n_points=30
                    )

                    if not pd_df.empty:
                        import plotly.express as px

                        fig = px.line(
                            pd_df,
                            x="feature_value",
                            y="prediction",
                            title=f"Partial Dependence: {selected_feature} ({selected_model})",
                            labels={
                                "feature_value": selected_feature,
                                "prediction": "Predicted Water Stress (%)",
                            },
                        )

                        fig.update_layout(template="plotly_white", height=400)
                        st.plotly_chart(fig, width="stretch")

                except Exception as e:
                    st.warning(f"Could not generate partial dependence: {e}")

        st.markdown("---")

        # Feature relationships section
        st.subheader("🔗 Feature Relationships")

        with st.expander("Correlation with Target"):
            try:
                target_col = "Level of water stress: freshwater withdrawal as a proportion of available freshwater resources"
                feature_cols = get_feature_columns(df_historical)

                correlations = []
                for feature in feature_cols:
                    try:
                        valid_idx = (
                            df_historical[feature].notna() & df_historical[target_col].notna()
                        )
                        if valid_idx.sum() > 0:
                            corr = df_historical.loc[valid_idx, feature].corr(
                                df_historical.loc[valid_idx, target_col]
                            )
                            correlations.append(
                                {
                                    "feature": feature,
                                    "correlation": corr,
                                    "abs_correlation": abs(corr),
                                }
                            )
                    except Exception:
                        pass

                if correlations:
                    corr_df = pd.DataFrame(correlations).sort_values(
                        "abs_correlation", ascending=True
                    )

                    import plotly.express as px

                    fig = px.bar(
                        corr_df,
                        x="correlation",
                        y="feature",
                        orientation="h",
                        title="Feature Correlation with Water Stress",
                        color="correlation",
                        color_continuous_scale="RdBu",
                        labels={"correlation": "Pearson Correlation", "feature": "Feature"},
                    )

                    fig.update_layout(
                        height=max(400, len(feature_cols) * 20), template="plotly_white"
                    )
                    st.plotly_chart(fig, width="stretch")

            except Exception as e:
                st.warning(f"Could not calculate correlations: {e}")

        st.markdown("---")

        # About features
        st.subheader("ℹ️ About These Features")

        with st.expander("Feature Descriptions"):
            st.markdown(
                """
            The dashboard uses 5 key features from World Bank environmental indicators:

            - **Water Availability**: Annual renewable water resources
            - **Population Pressure**: Population density and growth
            - **Agricultural Demand**: Agricultural value added and land use
            - **Economic Factors**: GDP and industrial activity
            - **Environmental Indicators**: Forest coverage and precipitation

            These features are selected to capture the main drivers of water stress in Tunisia.
            """
            )

    except Exception as e:
        st.error(f"Error loading feature analysis: {e}")


if __name__ == "__main__":
    run()
