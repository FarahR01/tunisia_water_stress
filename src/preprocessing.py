"""Data preprocessing utilities for water stress analysis.

Provides functions for handling missing values, feature selection, and data cleaning.
"""

from typing import List

import pandas as pd


def drop_sparse_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns with more than `threshold` fraction of missing values.

    Args:
        df: Input DataFrame to clean.
        threshold: Maximum allowed fraction of missing values (0.0 to 1.0).
                   Default is 0.5 (50%).

    Returns:
        pd.DataFrame: DataFrame with sparse columns removed.
    """
    frac_missing = df.isna().mean()
    keep = frac_missing[frac_missing <= threshold].index.tolist()
    return df.loc[:, keep]


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate linearly over time and then forward/back-fill remaining values.

    Handles missing values by:
    1. Linear interpolation along the year index (axis 0)
    2. Forward-fill followed by back-fill for any remaining NaN values

    Args:
        df: Input DataFrame with time series data indexed by year.

    Returns:
        pd.DataFrame: DataFrame with filled missing values.
    """
    df = df.copy()
    # interpolate along the year index
    df = df.interpolate(method="linear", axis=0, limit_direction="both")
    df = df.ffill().bfill()
    return df


def select_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Return df with only columns that exist in feature_names (order preserved).

    Args:
        df: Input DataFrame.
        feature_names: List of column names to select.

    Returns:
        pd.DataFrame: DataFrame with only selected features that exist in df.
    """
    available = [c for c in feature_names if c in df.columns]
    return df.loc[:, available]
