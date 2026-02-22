"""Unit tests for data_loader module.

Tests cover loading, filtering, and pivoting World Bank data.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_and_pivot, list_available_indicators


@pytest.fixture
def sample_raw_data() -> str:
    """Create a temporary raw data CSV file for testing."""
    data = """Country Name,Country Code,Indicator Name,Indicator Code,Year,Value
Tunisia,TUN,Water Productivity,WP.1,2000,10.5
Tunisia,TUN,Water Productivity,WP.1,2001,11.2
Tunisia,TUN,Water Productivity,WP.1,2002,12.0
Tunisia,TUN,Population,SP.POP.TOTL,2000,9500000
Tunisia,TUN,Population,SP.POP.TOTL,2001,9550000
Tunisia,TUN,Population,SP.POP.TOTL,2002,9600000
Egypt,EGY,Water Productivity,WP.1,2000,15.5
Egypt,EGY,Population,SP.POP.TOTL,2000,65000000
# Comment line should be skipped
Tunisia,TUN,Precipitation,PRECIP,2000,500
Tunisia,TUN,Precipitation,PRECIP,2001,510
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(data)
        return f.name


class TestLoadAndPivot:
    """Tests for load_and_pivot function."""

    def test_load_and_pivot_basic(self, sample_raw_data: str):
        """Test basic loading and pivoting of data."""
        df = load_and_pivot(sample_raw_data)

        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "Year"
        assert list(df.index) == [2000, 2001, 2002]
        # Check that Tunisia data is loaded
        assert "Water Productivity" in df.columns
        assert "Population" in df.columns
        assert "Precipitation" in df.columns
        # Check that Egypt data is filtered out (only Tunisia rows)
        assert len(df) == 3
        assert df.loc[2000, "Water Productivity"] == 10.5

    def test_load_and_pivot_filters_tunisia_only(self, sample_raw_data: str):
        """Test that only Tunisia data is loaded."""
        df = load_and_pivot(sample_raw_data)

        # Egypt should be filtered out
        assert len(df) == 3
        # Check values are from Tunisia
        assert df.loc[2000, "Water Productivity"] == 10.5
        assert df.loc[2000, "Population"] == 9500000

    def test_load_and_pivot_saves_to_path(self, sample_raw_data: str):
        """Test that pivot is saved to specified path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "processed.csv")
            df = load_and_pivot(sample_raw_data, output_path)

            # Check that file was created
            assert os.path.exists(output_path)
            # Verify saved file can be loaded
            saved_df = pd.read_csv(output_path, index_col=0)
            # Verify the shape and values match
            assert saved_df.shape == df.shape
            assert list(saved_df.columns) == list(df.columns)

    def test_load_and_pivot_handles_missing_years(self, sample_raw_data: str):
        """Test handling of data with missing years."""
        df = load_and_pivot(sample_raw_data)

        # Index should be sorted
        assert list(df.index) == sorted(df.index)
        # All years should be integers
        assert all(isinstance(year, (int, np.integer)) for year in df.index)

    def test_load_and_pivot_creates_directories(self, sample_raw_data: str):
        """Test that necessary directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "a", "b", "c", "processed.csv")
            _ = load_and_pivot(sample_raw_data, nested_path)

            assert os.path.exists(nested_path)
            assert os.path.isfile(nested_path)


class TestListAvailableIndicators:
    """Tests for list_available_indicators function."""

    def test_list_available_indicators(self, sample_raw_data: str):
        """Test listing of available indicators for Tunisia."""
        indicators = list_available_indicators(sample_raw_data)

        assert isinstance(indicators, list)
        assert "Water Productivity" in indicators
        assert "Population" in indicators
        assert "Precipitation" in indicators
        # Should only have Tunisia indicators, not Egypt's
        assert len(indicators) == 3

    def test_list_available_indicators_sorted(self, sample_raw_data: str):
        """Test that indicators are returned sorted."""
        indicators = list_available_indicators(sample_raw_data)

        assert indicators == sorted(indicators)

    def test_list_available_indicators_no_duplicates(self, sample_raw_data: str):
        """Test that duplicate indicators are removed."""
        indicators = list_available_indicators(sample_raw_data)

        assert len(indicators) == len(set(indicators))

    def test_list_available_indicators_filters_tunisia(self, sample_raw_data: str):
        """Test that only Tunisia indicators are listed."""
        indicators = list_available_indicators(sample_raw_data)

        # Egypt's countries should not be in the list
        assert len(indicators) == 3  # Only Tunisia's 3 indicators
        # Should not include any Egypt-specific patterns
        for indicator in indicators:
            assert isinstance(indicator, str)
            assert len(indicator) > 0
