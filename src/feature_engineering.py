"""Feature engineering utilities for water stress prediction models.

Provides functions for creating time-lagged features and adding temporal information.
"""

from typing import List

import pandas as pd


def add_lag_features(df: pd.DataFrame, columns: List[str], lags: int = 1) -> pd.DataFrame:
    """Add lagged versions of specified columns (lag 1..lags).

    Creates new columns with lagged values. For example, with lags=2, creates
    'col_lag1' and 'col_lag2' columns. Rows with insufficient history are dropped.

    Args:
        df: Input DataFrame indexed by year or time.
        columns: List of column names to add lags for.
        lags: Number of lag periods to create. Default is 1.

    Returns:
        pd.DataFrame: DataFrame with new lag columns and rows with NaN removed.
    """
    out = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for lag in range(1, lags + 1):
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    out = out.dropna()
    return out


def add_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'year' column extracted from the DataFrame index.

    Args:
        df: Input DataFrame indexed by year (as integer or numeric type).

    Returns:
        pd.DataFrame: DataFrame with new 'year' column containing year values as integers.
    """
    out = df.copy()
    out["year"] = out.index.astype(int)
    return out
