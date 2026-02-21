import os
import pandas as pd


def load_and_pivot(raw_csv_path: str, processed_path: str = None) -> pd.DataFrame:
    """Load World Bank long-form CSV, filter Tunisia and pivot to years x indicators.

    Skips comment lines beginning with '#'. Returns a DataFrame indexed by Year (int).
    If processed_path is provided the result is saved there as CSV.
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


def list_available_indicators(raw_csv_path: str) -> list:
    """Return sorted list of indicator names present for Tunisia in the raw CSV."""
    df = pd.read_csv(raw_csv_path, comment="#")
    df = df[df["Country Name"].str.strip().str.lower() == "tunisia"]
    inds = sorted(df["Indicator Name"].unique().tolist())
    return inds
