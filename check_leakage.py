import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from data_loader import load_and_pivot

# Load the processed data
raw_path = os.path.join("data", "raw", "environment_tun.csv")
processed_path = os.path.join("data", "processed", "processed_tunisia.csv")

if not os.path.exists(processed_path):
    df = load_and_pivot(raw_path, processed_path)
else:
    df = pd.read_csv(processed_path, index_col=0)
    df.index = df.index.astype(int)

# Target indicator
target = "Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)"

# Selected features (from the training output)
features = [
    'Annual freshwater withdrawals, domestic (% of total freshwater withdrawal)',
    'Annual freshwater withdrawals, industry (% of total freshwater withdrawal)',
    'Annual freshwater withdrawals, total (% of internal resources)',
    'Annual freshwater withdrawals, total (billion cubic meters)',
]

# Check if all columns exist
print(f"Target: {target}")
print(f"Target in data: {target in df.columns}\n")

for feat in features:
    if feat in df.columns:
        # Calculate correlation
        corr = df[[target, feat]].corr().iloc[0, 1]
        print(f"Feature: {feat}")
        print(f"  Correlation with target: {corr:.6f}")
        print()
    else:
        print(f"Feature NOT in data: {feat}\n")

# Check all columns for extremely high correlation with target
print("\n=== All columns with correlation >= 0.95 with target ===")
all_corr = df.corr()[target].sort_values(ascending=False)
high_corr = all_corr[all_corr.abs() >= 0.95]
for idx, val in high_corr.items():
    print(f"{idx}: {val:.6f}")
