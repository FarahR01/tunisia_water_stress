"""Integration tests for the complete pipeline.

Tests the full workflow from data loading through model training and evaluation.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.preprocessing import (
    drop_sparse_columns,
    fill_missing,
    select_features,
)
from src.feature_engineering import add_lag_features, add_year_column
from src.evaluate import regression_metrics, plot_actual_vs_pred


@pytest.fixture
def sample_processed_data() -> str:
    """Create a sample processed data CSV for integration testing."""
    # Simulating processed data output
    years = list(range(1990, 2020))
    data = {
        "indicator1": np.linspace(100, 150, len(years)),
        "indicator2": np.linspace(50, 80, len(years)),
        "indicator3": np.linspace(200, 250, len(years)),
        "water_stress": np.linspace(20, 60, len(years)),
    }

    df = pd.DataFrame(data, index=years)
    df.index.name = "Year"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name)
        return f.name


class TestPipelineDataLoad:
    """Test data loading stage of pipeline."""

    def test_load_and_clean_data(self, sample_processed_data: str):
        """Test loading and cleaning data."""
        df = pd.read_csv(sample_processed_data, index_col="Year")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 30
        assert df.shape[1] == 4


class TestPipelinePreprocessing:
    """Test preprocessing stage of pipeline."""

    def test_preprocess_pipeline(self, sample_processed_data: str):
        """Test preprocessing pipeline with missing values."""
        # Load data
        df = pd.read_csv(sample_processed_data, index_col="Year")

        # Add some missing values
        df.loc[1995:2000, "indicator1"] = np.nan

        # Preprocess
        df_clean = drop_sparse_columns(df, threshold=0.3)
        df_filled = fill_missing(df_clean)

        assert df_filled.isna().sum().sum() == 0
        assert len(df_filled) > 0

    def test_feature_selection_pipeline(self, sample_processed_data: str):
        """Test feature selection in pipeline."""
        df = pd.read_csv(sample_processed_data, index_col="Year")

        features = ["indicator1", "indicator2"]
        df_selected = select_features(df, features)

        assert len(df_selected.columns) == 2
        assert "water_stress" not in df_selected.columns


class TestPipelineFeatureEngineering:
    """Test feature engineering stage of pipeline."""

    def test_feature_engineering_pipeline(self, sample_processed_data: str):
        """Test complete feature engineering pipeline."""
        df = pd.read_csv(sample_processed_data, index_col="Year")

        # Add lags
        df_lagged = add_lag_features(df, columns=["indicator1", "indicator2"], lags=2)

        # Add year
        df_with_year = add_year_column(df_lagged)

        assert "indicator1_lag1" in df_with_year.columns
        assert "indicator1_lag2" in df_with_year.columns
        assert "year" in df_with_year.columns
        assert len(df_with_year) < len(df)  # Some rows dropped due to lag


class TestPipelineTrainEvaluate:
    """Test training and evaluation stage of pipeline."""

    def test_train_evaluate_pipeline(self, sample_processed_data: str):
        """Test complete train-evaluate pipeline."""
        # Load data
        df = pd.read_csv(sample_processed_data, index_col="Year")

        # Prepare features and target
        features = ["indicator1", "indicator2", "indicator3"]
        X = df[features].values
        y = df["water_stress"].values

        # Train-test split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        metrics_train = regression_metrics(y_train, y_pred_train)
        metrics_test = regression_metrics(y_test, y_pred_test)

        assert "MAE" in metrics_train
        assert "RMSE" in metrics_train
        assert "R2" in metrics_train
        assert all(isinstance(v, float) for v in metrics_train.values())
        assert all(isinstance(v, float) for v in metrics_test.values())

    def test_plot_generation(self, sample_processed_data: str):
        """Test plot generation during evaluation."""
        df = pd.read_csv(sample_processed_data, index_col="Year")

        # Simple predictions
        years = df.index.values
        y_true = df["water_stress"].values
        y_pred = y_true * 1.1  # Simple prediction

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = os.path.join(tmpdir, "plot.png")
            plot_actual_vs_pred(years, y_true, y_pred, plot_path)

            assert os.path.exists(plot_path)
            assert os.path.getsize(plot_path) > 0


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""

    def test_full_pipeline_workflow(self, sample_processed_data: str):
        """Test complete pipeline from data to prediction."""
        # 1. Load data
        df = pd.read_csv(sample_processed_data, index_col="Year")

        # 2. Preprocess
        df_clean = drop_sparse_columns(df, threshold=0.5)
        df_filled = fill_missing(df_clean)

        # 3. Feature engineering
        feature_columns = [col for col in df_filled.columns if col != "water_stress"]
        df_lagged = add_lag_features(df_filled, columns=feature_columns, lags=1)
        df_with_year = add_year_column(df_lagged)

        # 4. Prepare for modeling
        feature_cols = [col for col in df_with_year.columns if col != "water_stress"]
        X = df_with_year[feature_cols].values
        y = df_with_year["water_stress"].values

        # 5. Train model
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = LinearRegression()
        model.fit(X_train, y_train)

        # 6. Evaluate
        y_pred = model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)

        # 7. Verify results
        assert len(y_pred) == len(y_test)
        assert "MAE" in metrics
        assert metrics["MAE"] > 0
        assert -1 <= metrics["R2"] <= 1

        # 8. Save outputs
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = os.path.join(tmpdir, "predictions.png")
            plot_actual_vs_pred(df_with_year.index[: len(y_test)], y_test, y_pred, plot_path)
            assert os.path.exists(plot_path)

    def test_pipeline_reproducibility(self, sample_processed_data: str):
        """Test that pipeline produces consistent results."""
        # Run pipeline twice and verify consistency
        results = []

        for _ in range(2):
            df = pd.read_csv(sample_processed_data, index_col="Year")
            df_filled = fill_missing(df)

            feature_cols = [col for col in df_filled.columns if col != "water_stress"]
            X = df_filled[feature_cols].values
            y = df_filled["water_stress"].values

            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = regression_metrics(y_test, y_pred)
            results.append(metrics)

        # Results should be identical
        assert results[0]["MAE"] == results[1]["MAE"]
        assert results[0]["RMSE"] == results[1]["RMSE"]
        assert results[0]["R2"] == results[1]["R2"]
