"""Reusable Streamlit visualization components."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Color scheme
PRIMARY_COLOR = "#0F766E"
SECONDARY_COLOR = "#06B6D4"
SUCCESS_COLOR = "#10B981"
WARNING_COLOR = "#F59E0B"
DANGER_COLOR = "#EF4444"


def metric_card(col, label: str, value, delta=None, delta_color: str = "normal"):
    """Display a metric card in a column."""
    with col:
        if isinstance(value, str):
            st.metric(label, value)
        elif delta is not None:
            st.metric(label, f"{value:.2f}", delta=f"{delta:+.2f}", delta_color=delta_color)
        else:
            st.metric(label, f"{value:.2f}")


def plot_historical_trend(
    df: pd.DataFrame, column: str, title: str = None, show_ma: bool = True, ma_window: int = 5
):
    """Plot historical data with optional moving average."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Year"],
            y=df[column],
            mode="lines+markers",
            name="Historical Data",
            line=dict(color=PRIMARY_COLOR, width=3),
            marker=dict(size=6, color=PRIMARY_COLOR),
            hovertemplate="<b>Year: %{x}</b><br>Water Stress: %{y:.2f}%<extra></extra>",
        )
    )

    if show_ma and len(df) > ma_window:
        ma = df[column].rolling(window=ma_window).mean()
        fig.add_trace(
            go.Scatter(
                x=df["Year"],
                y=ma,
                mode="lines",
                name=f"{ma_window}-Year Moving Average",
                line=dict(color=SUCCESS_COLOR, width=2, dash="dash"),
                hovertemplate="<b>Year: %{x}</b><br>MA: %{y:.2f}%<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title or f"{column} Over Time", font=dict(size=18, color=PRIMARY_COLOR)),
        xaxis_title="Year",
        yaxis_title=column,
        hovermode="x unified",
        template="plotly_white",
        height=450,
        font=dict(size=11),
        plot_bgcolor="rgba(245, 248, 250, 0.5)",
        margin=dict(l=60, r=40, t=60, b=60),
    )

    return fig


def plot_model_comparison(df: pd.DataFrame):
    """Plot comparison of model predictions."""
    if df.empty or "prediction_2030" not in df.columns:
        return None

    fig = px.bar(
        df.sort_values("prediction_2030", ascending=True),
        x="prediction_2030",
        y="model",
        orientation="h",
        title="2030 Water Stress Predictions by Model",
        labels={"prediction_2030": "Predicted Water Stress (%)", "model": "Model"},
        color="prediction_2030",
        color_continuous_scale=[(0, PRIMARY_COLOR), (1, SECONDARY_COLOR)],
    )

    fig.update_traces(hovertemplate="<b>%{y}</b><br>Prediction: %{x:.2f}%<extra></extra>")

    fig.update_layout(
        height=400,
        template="plotly_white",
        showlegend=False,
        title=dict(font=dict(size=16, color=PRIMARY_COLOR)),
        xaxis_title_font=dict(size=12),
        yaxis_title_font=dict(size=12),
        plot_bgcolor="rgba(245, 248, 250, 0.5)",
        margin=dict(l=120, r=40, t=60, b=60),
    )

    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 10):
    """Plot top N important features."""
    if importance_df.empty:
        return None

    top_features = importance_df.head(top_n)

    fig = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {top_n} Feature Importance",
        labels={"importance": "Importance Score", "feature": "Feature"},
        color="importance",
        color_continuous_scale=[(0, SECONDARY_COLOR), (1, PRIMARY_COLOR)],
    )

    fig.update_traces(hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>")

    fig.update_layout(
        height=max(400, top_n * 35),
        template="plotly_white",
        yaxis={"categoryorder": "total ascending"},
        showlegend=False,
        title=dict(font=dict(size=16, color=PRIMARY_COLOR)),
        plot_bgcolor="rgba(245, 248, 250, 0.5)",
        margin=dict(l=200, r=40, t=60, b=60),
    )

    return fig


def plot_scenario_comparison(scenarios: Dict[str, pd.DataFrame]):
    """Plot multiple prediction scenarios."""
    fig = go.Figure()

    colors = [PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR]

    for idx, (scenario_name, df) in enumerate(scenarios.items()):
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=df["year"],
                y=df["prediction"],
                mode="lines+markers",
                name=scenario_name,
                line=dict(color=color, width=3),
                marker=dict(size=7, color=color),
                hovertemplate="<b>"
                + scenario_name
                + "</b><br>Year: %{x}<br>Stress: %{y:.2f}%<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="Water Stress Scenarios (2025-2040)", font=dict(size=18, color=PRIMARY_COLOR)
        ),
        xaxis_title="Year",
        yaxis_title="Water Stress Level (%)",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        plot_bgcolor="rgba(245, 248, 250, 0.5)",
        margin=dict(l=60, r=40, t=60, b=60),
    )

    return fig


def plot_uncertainty_band(
    df_mean: pd.DataFrame, lower_bound: pd.DataFrame, upper_bound: pd.DataFrame
):
    """Plot predictions with uncertainty bands."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_mean["year"].tolist() + df_mean["year"].tolist()[::-1],
            y=upper_bound["prediction"].tolist() + lower_bound["prediction"].tolist()[::-1],
            fill="toself",
            fillcolor=f"rgba(15, 118, 110, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Confidence Interval",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_mean["year"],
            y=df_mean["prediction"],
            mode="lines+markers",
            name="Mean Prediction",
            line=dict(color=PRIMARY_COLOR, width=3),
            marker=dict(size=7, color=PRIMARY_COLOR),
            hovertemplate="<b>Year: %{x}</b><br>Prediction: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Water Stress Prediction with Uncertainty", font=dict(size=18, color=PRIMARY_COLOR)
        ),
        xaxis_title="Year",
        yaxis_title="Water Stress Level (%)",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        plot_bgcolor="rgba(245, 248, 250, 0.5)",
        margin=dict(l=60, r=40, t=60, b=60),
    )

    return fig


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Feature Correlations"):
    """Plot correlation heatmap with improved readability."""
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return None

    # Select relevant columns (exclude Year)
    cols_to_corr = [col for col in numeric_df.columns if col != "Year"]
    corr_matrix = numeric_df[cols_to_corr].corr()

    # Shorten long column names for display
    display_cols = [col[:25] + "..." if len(col) > 25 else col for col in corr_matrix.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=display_cols,
            y=display_cols,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 11, "color": "black"},
            colorbar=dict(title="Correlation<br>Coefficient", thickness=15, len=0.7),
            hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=PRIMARY_COLOR)),
        height=600,
        width=700,
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        template="plotly_white",
        margin=dict(l=200, b=200, r=100, t=100),
        plot_bgcolor="white",
    )

    return fig


def metric_summary_cards(metrics_df: pd.DataFrame, col_count: int = 4):
    """Display metrics as summary cards."""
    if metrics_df.empty:
        return

    cols = st.columns(col_count)

    for idx, (_, row) in enumerate(metrics_df.iterrows()):
        with cols[idx % col_count]:
            st.metric(row.get("metric", "Metric"), f"{row.get('value', 0):.4f}")


def comparison_table(df: pd.DataFrame, title: Optional[str] = None):
    """Display a formatted comparison table."""
    if title:
        st.subheader(title)

    st.dataframe(df, use_container_width=True)
