"""Unit tests for preprocessing module.

Tests cover missing value handling, feature selection, and data cleaning.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    drop_sparse_columns,
    fill_missing,
    select_features,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame with some missing values."""
    return pd.DataFrame(
        {
            "feature1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "feature2": [np.nan, np.nan, np.nan, np.nan, np.nan],  # All missing
            "feature3": [1.0, 2.0, 3.0, np.nan, 5.0],
            "feature4": [10.0, 20.0, 30.0, 40.0, 50.0],  # No missing
            "feature5": [np.nan, 2.0, np.nan, np.nan, 5.0],  # 60% missing
        }
    )


@pytest.fixture
def time_series_dataframe() -> pd.DataFrame:
    """Create a time series DataFrame indexed by year."""
    years = [2000, 2001, 2002, 2003, 2004]
    df = pd.DataFrame(
        {
            "temperature": [20.0, np.nan, 22.0, 23.0, np.nan],
            "precipitation": [100.0, 110.0, np.nan, 130.0, 140.0],
            "humidity": [np.nan, np.nan, 65.0, 70.0, 75.0],
        },
        index=years,
    )
    df.index.name = "Year"
    return df


class TestDropSparseColumns:
    """Tests for drop_sparse_columns function."""

    def test_drop_sparse_columns_removes_all_nan(self, sample_dataframe: pd.DataFrame):
        """Test that columns with all NaN values are removed."""
        result = drop_sparse_columns(sample_dataframe, threshold=0.5)

        assert "feature2" not in result.columns  # All missing (100%)
        assert "feature1" in result.columns
        assert "feature3" in result.columns
        assert "feature4" in result.columns

    def test_drop_sparse_columns_threshold(self, sample_dataframe: pd.DataFrame):
        """Test threshold parameter controls sparsity cutoff."""
        # With threshold=0.5, keep columns with <= 50% missing
        result_50 = drop_sparse_columns(sample_dataframe, threshold=0.5)
        # feature5 has 60% missing, should be removed
        assert "feature5" not in result_50.columns

        # With threshold=0.7, keep columns with <= 70% missing
        result_70 = drop_sparse_columns(sample_dataframe, threshold=0.7)
        # feature5 has 60% missing, should be kept
        assert "feature5" in result_70.columns

    def test_drop_sparse_columns_preserves_data(self, sample_dataframe: pd.DataFrame):
        """Test that remaining data is unchanged."""
        result = drop_sparse_columns(sample_dataframe, threshold=0.5)

        # Check that the data in remaining columns is unchanged
        assert result["feature1"].iloc[0] == sample_dataframe["feature1"].iloc[0]
        assert result["feature4"].iloc[2] == sample_dataframe["feature4"].iloc[2]

    def test_drop_sparse_columns_default_threshold(self, sample_dataframe: pd.DataFrame):
        """Test default threshold value."""
        result = drop_sparse_columns(sample_dataframe)  # Default threshold=0.5

        # Should match explicit threshold=0.5
        assert set(result.columns) == {
            "feature1",
            "feature3",
            "feature4",
        }

    def test_drop_sparse_columns_returns_dataframe(self, sample_dataframe: pd.DataFrame):
        """Test that result is a DataFrame."""
        result = drop_sparse_columns(sample_dataframe)

        assert isinstance(result, pd.DataFrame)


class TestFillMissing:
    """Tests for fill_missing function."""

    def test_fill_missing_basic(self, time_series_dataframe: pd.DataFrame):
        """Test basic missing value filling."""
        result = fill_missing(time_series_dataframe)

        # Should have no NaN values
        assert result.isna().sum().sum() == 0

    def test_fill_missing_preserves_index(self, time_series_dataframe: pd.DataFrame):
        """Test that the index is preserved."""
        result = fill_missing(time_series_dataframe)

        assert list(result.index) == list(time_series_dataframe.index)
        assert result.index.name == "Year"

    def test_fill_missing_preserves_shape(self, time_series_dataframe: pd.DataFrame):
        """Test that shape is preserved."""
        result = fill_missing(time_series_dataframe)

        assert result.shape == time_series_dataframe.shape

    def test_fill_missing_interpolates_linearly(self, time_series_dataframe: pd.DataFrame):
        """Test that interpolation works correctly."""
        result = fill_missing(time_series_dataframe)

        # For temperature: [20.0, NaN, 22.0, 23.0, NaN]
        # Should interpolate to: [20.0, 21.0, 22.0, 23.0, 23.0] (forward fill for end)
        assert result.loc[2001, "temperature"] == pytest.approx(21.0)

    def test_fill_missing_doesnt_modify_original(self, time_series_dataframe: pd.DataFrame):
        """Test that the original DataFrame is not modified."""
        original_copy = time_series_dataframe.copy()
        fill_missing(time_series_dataframe)

        pd.testing.assert_frame_equal(time_series_dataframe, original_copy)

    def test_fill_missing_returns_dataframe(self, time_series_dataframe: pd.DataFrame):
        """Test that result is a DataFrame."""
        result = fill_missing(time_series_dataframe)

        assert isinstance(result, pd.DataFrame)


class TestSelectFeatures:
    """Tests for select_features function."""

    def test_select_features_basic(self, sample_dataframe: pd.DataFrame):
        """Test basic feature selection."""
        features_to_select = ["feature1", "feature4"]
        result = select_features(sample_dataframe, features_to_select)

        assert list(result.columns) == features_to_select

    def test_select_features_preserves_order(self, sample_dataframe: pd.DataFrame):
        """Test that column order is preserved."""
        # Try non-alphabetical order
        features_to_select = ["feature4", "feature1", "feature3"]
        result = select_features(sample_dataframe, features_to_select)

        assert list(result.columns) == features_to_select

    def test_select_features_ignores_missing_columns(self, sample_dataframe: pd.DataFrame):
        """Test that non-existent columns are silently ignored."""
        features_to_select = ["feature1", "feature99", "feature4"]
        result = select_features(sample_dataframe, features_to_select)

        # Should only include features that exist
        assert list(result.columns) == ["feature1", "feature4"]
        assert "feature99" not in result.columns

    def test_select_features_preserves_data(self, sample_dataframe: pd.DataFrame):
        """Test that data is unchanged."""
        features_to_select = ["feature1", "feature4"]
        result = select_features(sample_dataframe, features_to_select)

        pd.testing.assert_frame_equal(result, sample_dataframe[features_to_select])

    def test_select_features_empty_list(self, sample_dataframe: pd.DataFrame):
        """Test with empty feature list."""
        result = select_features(sample_dataframe, [])

        assert len(result.columns) == 0
        assert len(result) == len(sample_dataframe)

    def test_select_features_returns_dataframe(self, sample_dataframe: pd.DataFrame):
        """Test that result is a DataFrame."""
        result = select_features(sample_dataframe, ["feature1"])

        assert isinstance(result, pd.DataFrame)
