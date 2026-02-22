import pandas as pd


def drop_sparse_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns with more than `threshold` fraction of missing values."""
    frac_missing = df.isna().mean()
    keep = frac_missing[frac_missing <= threshold].index.tolist()
    return df.loc[:, keep]


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate linearly over time and then forward/back-fill remaining values."""
    df = df.copy()
    # interpolate along the year index
    df = df.interpolate(method="linear", axis=0, limit_direction="both")
    df = df.ffill().bfill()
    return df


def select_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Return df with only columns that exist in feature_names (order preserved)."""
    available = [c for c in feature_names if c in df.columns]
    return df.loc[:, available]
