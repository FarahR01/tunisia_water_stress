"""Unit tests for feature_engineering module.

Tests cover lagged feature creation and temporal feature engineering.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import add_lag_features, add_year_column


@pytest.fixture
def sample_time_series() -> pd.DataFrame:
    """Create a sample time series DataFrame indexed by year."""
    years = [2000, 2001, 2002, 2003, 2004, 2005]
    df = pd.DataFrame(
        {
            "temperature": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
            "precipitation": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "humidity": [60.0, 65.0, 70.0, 75.0, 80.0, 85.0],
        },
        index=years,
    )
    df.index.name = "Year"
    return df


class TestAddLagFeatures:
    """Tests for add_lag_features function."""

    def test_add_lag_features_single_lag(self, sample_time_series: pd.DataFrame):
        """Test adding single lag features."""
        result = add_lag_features(sample_time_series, columns=["temperature"], lags=1)

        assert "temperature_lag1" in result.columns
        assert len(result) == 5  # Original 6 - 1 (first row with NaN dropped)

    def test_add_lag_features_multiple_lags(self, sample_time_series: pd.DataFrame):
        """Test adding multiple lag features."""
        result = add_lag_features(sample_time_series, columns=["temperature"], lags=3)

        assert "temperature_lag1" in result.columns
        assert "temperature_lag2" in result.columns
        assert "temperature_lag3" in result.columns
        assert len(result) == 3  # Original 6 - 3 (first 3 rows dropped)

    def test_add_lag_features_multiple_columns(self, sample_time_series: pd.DataFrame):
        """Test adding lags for multiple columns."""
        result = add_lag_features(
            sample_time_series, columns=["temperature", "precipitation"], lags=1
        )

        assert "temperature_lag1" in result.columns
        assert "precipitation_lag1" in result.columns
        assert "temperature" in result.columns
        assert "precipitation" in result.columns

    def test_add_lag_features_lag_values(self, sample_time_series: pd.DataFrame):
        """Test that lag values are correct."""
        result = add_lag_features(sample_time_series, columns=["temperature"], lags=1)

        # temperature_lag1 should be shifted by 1
        # Original: [20, 21, 22, 23, 24, 25]
        # Lag1: [NaN, 20, 21, 22, 23, 24] -> after dropna: [21, 22, 23, 24, 25]
        assert result["temperature_lag1"].iloc[0] == 20.0
        assert result["temperature_lag1"].iloc[1] == 21.0

    def test_add_lag_features_ignores_missing_columns(self, sample_time_series: pd.DataFrame):
        """Test that non-existent columns are silently ignored."""
        result = add_lag_features(
            sample_time_series, columns=["temperature", "nonexistent_column"], lags=1
        )

        assert "temperature_lag1" in result.columns
        assert "nonexistent_column_lag1" not in result.columns

    def test_add_lag_features_preserves_original_columns(self, sample_time_series: pd.DataFrame):
        """Test that original columns are preserved."""
        result = add_lag_features(sample_time_series, columns=["temperature"], lags=1)

        assert "temperature" in result.columns
        assert "precipitation" in result.columns
        # Humidity should also be preserved
        assert "humidity" in result.columns

    def test_add_lag_features_drops_na(self, sample_time_series: pd.DataFrame):
        """Test that rows with NaN are dropped."""
        original_len = len(sample_time_series)
        result = add_lag_features(sample_time_series, columns=["temperature"], lags=2)

        # Should drop first 2 rows (where lags have NaN)
        assert len(result) == original_len - 2

    def test_add_lag_features_returns_dataframe(self, sample_time_series: pd.DataFrame):
        """Test that result is a DataFrame."""
        result = add_lag_features(sample_time_series, columns=["temperature"], lags=1)

        assert isinstance(result, pd.DataFrame)

    def test_add_lag_features_default_lags(self, sample_time_series: pd.DataFrame):
        """Test default lag value."""
        result = add_lag_features(
            sample_time_series,
            columns=["temperature"],
            # lags defaults to 1
        )

        assert "temperature_lag1" in result.columns
        assert "temperature_lag2" not in result.columns

    def test_add_lag_features_doesnt_modify_original(self, sample_time_series: pd.DataFrame):
        """Test that the original DataFrame is not modified."""
        original_copy = sample_time_series.copy()
        add_lag_features(sample_time_series, columns=["temperature"], lags=1)

        pd.testing.assert_frame_equal(sample_time_series, original_copy)


class TestAddYearColumn:
    """Tests for add_year_column function."""

    def test_add_year_column_basic(self, sample_time_series: pd.DataFrame):
        """Test basic year column addition."""
        result = add_year_column(sample_time_series)

        assert "year" in result.columns
        assert list(result["year"]) == [2000, 2001, 2002, 2003, 2004, 2005]

    def test_add_year_column_year_values_are_integers(self, sample_time_series: pd.DataFrame):
        """Test that year values are integers."""
        result = add_year_column(sample_time_series)

        assert all(isinstance(year, (int, np.integer)) for year in result["year"])

    def test_add_year_column_preserves_original_columns(self, sample_time_series: pd.DataFrame):
        """Test that original columns are preserved."""
        result = add_year_column(sample_time_series)

        assert "temperature" in result.columns
        assert "precipitation" in result.columns
        assert "humidity" in result.columns

    def test_add_year_column_preserves_data(self, sample_time_series: pd.DataFrame):
        """Test that original data is unchanged."""
        result = add_year_column(sample_time_series)

        # Remove year column and compare
        result_without_year = result.drop(columns=["year"])
        pd.testing.assert_frame_equal(result_without_year, sample_time_series)

    def test_add_year_column_preserves_index(self, sample_time_series: pd.DataFrame):
        """Test that the index is preserved."""
        result = add_year_column(sample_time_series)

        assert list(result.index) == list(sample_time_series.index)

    def test_add_year_column_returns_dataframe(self, sample_time_series: pd.DataFrame):
        """Test that result is a DataFrame."""
        result = add_year_column(sample_time_series)

        assert isinstance(result, pd.DataFrame)

    def test_add_year_column_with_float_index(self):
        """Test year column with float index."""
        df = pd.DataFrame(
            {"value": [1.0, 2.0, 3.0]},
            index=[2000.0, 2001.0, 2002.0],
        )
        df.index.name = "Year"

        result = add_year_column(df)

        assert list(result["year"]) == [2000, 2001, 2002]

    def test_add_year_column_doesnt_modify_original(self, sample_time_series: pd.DataFrame):
        """Test that the original DataFrame is not modified."""
        original_copy = sample_time_series.copy()
        add_year_column(sample_time_series)

        pd.testing.assert_frame_equal(sample_time_series, original_copy)
