import os
import pandas as pd
import numpy as np

processed = os.path.join("data", "processed", "processed_tunisia.csv")
models_before = os.path.join("models_test_leakfiltered", "metrics.csv")
models_after = os.path.join("models_test_colfiltered", "metrics.csv")

# Keywords copied from train.py
DEFAULT_FEATURE_KEYWORDS = [
    "freshwater",
    "renewable",
    "withdraw",
    "population",
    "urban",
    "rural",
    "agricultural land",
    "arable",
    "precipitation",
    "forest",
    "greenhouse",
    "water productivity",
]


def choose_columns_by_keywords(columns, keywords, max_features=12):
    cols = []
    cols_lower = [c.lower() for c in columns]
    for kw in keywords:
        for i, c in enumerate(columns):
            if kw in cols_lower[i] and c not in cols:
                cols.append(c)
                if len(cols) >= max_features:
                    return cols
    return cols


def drop_sparse_columns(df, threshold=0.5):
    frac_missing = df.isna().mean()
    keep = frac_missing[frac_missing <= threshold].index.tolist()
    return df.loc[:, keep]


def fill_missing(df):
    df = df.copy()
    df = df.interpolate(method="linear", axis=0, limit_direction="both")
    df = df.ffill().bfill()
    return df


# load processed
if not os.path.exists(processed):
    raise SystemExit("Processed CSV not found: " + processed)

raw = pd.read_csv(processed, index_col=0)
raw.index = raw.index.astype(int)

# auto target detection (simple heuristic used earlier)
cols = raw.columns.tolist()
cols_lower = [c.lower() for c in cols]

def choose_target(columns):
    cols_lower = [c.lower() for c in columns]
    for i, c in enumerate(columns):
        name = cols_lower[i]
        if ("withdraw" in name and "fresh" in name) or ("water stress" in name) or ("freshwater" in name and "%" in name):
            return c
    for i, c in enumerate(columns):
        name = cols_lower[i]
        if "freshwater" in name or "withdraw" in name:
            return c
    return None

target = choose_target(cols)
selected_features = choose_columns_by_keywords(cols, DEFAULT_FEATURE_KEYWORDS, max_features=12)
cols_needed = [c for c in selected_features if c in raw.columns]
if target not in raw.columns:
    raise SystemExit("Target not present in processed data")
cols_needed = cols_needed + [target]

data = raw.loc[:, cols_needed].copy()

orig_feature_count = len([c for c in data.columns if c != target])
data = drop_sparse_columns(data, threshold=0.5)
data = fill_missing(data)

# leakage
leak_thresh = 0.99
corr = data.corr().abs()
leaking = []
if target in corr.columns:
    target_corr = corr[target].drop(labels=[target], errors="ignore")
    leaking = target_corr[target_corr >= leak_thresh].index.tolist()

# after leakage
data_after_leak = data.drop(columns=leaking) if leaking else data.copy()
features_after_leak = [c for c in data_after_leak.columns if c != target]

# collinearity
col_thresh = 0.95
to_drop = set()
if features_after_leak:
    corr_mat = data_after_leak[features_after_leak].corr().abs()
    pairs = []
    for i_idx, i in enumerate(corr_mat.index):
        for j_idx, j in enumerate(corr_mat.columns):
            if j_idx <= i_idx:
                continue
            pairs.append((i, j, float(corr_mat.iat[i_idx, j_idx])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for i, j, val in pairs:
        if val >= col_thresh:
            if i not in to_drop and j not in to_drop:
                to_drop.add(j)

# read metrics
metrics_before = None
metrics_after = None
if os.path.exists(models_before):
    metrics_before = pd.read_csv(models_before)
if os.path.exists(models_after):
    metrics_after = pd.read_csv(models_after)

# Write report
os.makedirs("reports", exist_ok=True)
report_path = os.path.join("reports", "collinearity_report.md")
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write("# Collinearity & Leakage Filtering Report\n\n")
    fh.write(f"**Processed file:** {processed}\n\n")
    fh.write(f"**Auto-detected target:** {target}\n\n")
    fh.write(f"**Selected features (initial):** {orig_feature_count}\n\n")
    fh.write(f"**Leakage threshold:** {leak_thresh} — features dropped: {len(leaking)}\n")
    if leaking:
        fh.write("\n- " + "\n- ".join(leaking) + "\n")
    fh.write(f"\n**Collinearity threshold:** {col_thresh} — features flagged to drop: {len(to_drop)}\n")
    if to_drop:
        fh.write("\n- " + "\n- ".join(sorted(to_drop)) + "\n")

    fh.write("\n## Metrics\n\n")
    fh.write("Metrics before (leakage-filtered):\n\n")
    if metrics_before is not None:
        fh.write(metrics_before.to_string(index=False))
    else:
        fh.write("(metrics file not found)\n")
    fh.write("\n\nMetrics after (collinearity-filtered):\n\n")
    if metrics_after is not None:
        fh.write(metrics_after.to_string(index=False))
    else:
        fh.write("(metrics file not found)\n")

print("Report written to:", report_path)
