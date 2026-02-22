import pandas as pd


def add_lag_features(df: pd.DataFrame, columns: list, lags: int = 1) -> pd.DataFrame:
    """Add lagged versions of specified columns (lag 1..lags)."""
    out = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for lag in range(1, lags + 1):
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    out = out.dropna()
    return out


def add_year_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out.index.astype(int)
    return out
