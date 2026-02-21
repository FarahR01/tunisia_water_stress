import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from data_loader import load_and_pivot

# Load data
raw_path = os.path.join("data", "raw", "environment_tun.csv")
processed_path = os.path.join("data", "processed", "processed_tunisia.csv")

if not os.path.exists(processed_path):
    df = load_and_pivot(raw_path, processed_path)
else:
    df = pd.read_csv(processed_path, index_col=0)
    df.index = df.index.astype(int)

target = "Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)"

# Get full correlation matrix
print("=== TOP 20 STRONGEST CORRELATIONS WITH TARGET ===\n")
corr_with_target = df.corr()[target].abs().sort_values(ascending=False)
print(corr_with_target.head(20))

print("\n=== CHECKING FOR PERFECT DUPLICATES ===\n")
print(f"Target shape: {df[target].shape}")
print(f"Target mean: {df[target].mean():.6f}")
print(f"Target std: {df[target].std():.6f}")

# Check if any other column is identical or near-identical to target
for col in df.columns:
    if col == target:
        continue
    if df[col].equals(df[target]):
        print(f"EXACT DUPLICATE: {col}")
    elif (df[col] - df[target]).abs().max() < 1e-10:
        print(f"NEAR-DUPLICATE: {col} (max diff: {(df[col] - df[target]).abs().max()})")

print("\n=== SAMPLE OF TARGET VALUES ===")
print(df[target].head(10))
