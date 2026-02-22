"""Data loading utilities for Tunisia water stress analysis.

Provides functions to load World Bank data and extract indicator information.
"""

import os
from typing import List, Optional

import pandas as pd


def load_and_pivot(raw_csv_path: str, processed_path: Optional[str] = None) -> pd.DataFrame:
    """Load World Bank long-form CSV, filter Tunisia and pivot to years x indicators.

    Skips comment lines beginning with '#'. Returns a DataFrame indexed by Year (int).
    If processed_path is provided the result is saved there as CSV.

    Args:
        raw_csv_path: Path to the raw CSV file containing World Bank data.
        processed_path: Optional path to save the pivoted DataFrame as CSV.

    Returns:
        pd.DataFrame: DataFrame with years as index and indicators as columns.
    """
    df = pd.read_csv(raw_csv_path, comment="#")
    df = df[df["Country Name"].str.strip().str.lower() == "tunisia"]
    # Ensure Year is numeric
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])  # drop rows without year
    df["Year"] = df["Year"].astype(int)

    pivot = df.pivot_table(index="Year", columns="Indicator Name", values="Value", aggfunc="first")
    pivot.sort_index(inplace=True)

    if processed_path:
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        pivot.to_csv(processed_path)

    return pivot


def list_available_indicators(raw_csv_path: str) -> List[str]:
    """Return sorted list of indicator names present for Tunisia in the raw CSV.

    Args:
        raw_csv_path: Path to the raw CSV file containing World Bank data.

    Returns:
        List[str]: Sorted list of unique indicator names for Tunisia.
    """
    df = pd.read_csv(raw_csv_path, comment="#")
    df = df[df["Country Name"].str.strip().str.lower() == "tunisia"]
    inds = sorted(df["Indicator Name"].unique().tolist())
    return inds
