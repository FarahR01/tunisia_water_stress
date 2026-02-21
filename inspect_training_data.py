import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from data_loader import load_and_pivot
from preprocessing import drop_sparse_columns, fill_missing
from feature_engineering import add_lag_features, add_year_column

# Load and prepare data just like in train.py
raw_path = os.path.join("data", "raw", "environment_tun.csv")
processed_path = os.path.join("data", "processed", "processed_tunisia.csv")

if not os.path.exists(processed_path):
    df = load_and_pivot(raw_path, processed_path)
else:
    df = pd.read_csv(processed_path, index_col=0)
    df.index = df.index.astype(int)

target = "Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)"

features = [
    'Annual freshwater withdrawals, domestic (% of total freshwater withdrawal)',
    'Annual freshwater withdrawals, industry (% of total freshwater withdrawal)',
    'Annual freshwater withdrawals, total (% of internal resources)',
    'Annual freshwater withdrawals, total (billion cubic meters)',
    'Level of water stress: freshwater withdrawal as a proportion of available freshwater resources',
    'Renewable internal freshwater resources per capita (cubic meters)',
    'Renewable internal freshwater resources, total (billion cubic meters)',
    'Water productivity, total (constant 2015 US$ GDP per cubic meter of total freshwater withdrawal)',
    'Electricity production from renewable sources, excluding hydroelectric (% of total)',
    'Electricity production from renewable sources, excluding hydroelectric (kWh)',
    'Renewable electricity output (% of total electricity output)'
]

# Prepare data like train.py does
data = df.copy()
cols_needed = [c for c in features if c in data.columns] + [target]
data = data.loc[:, cols_needed]

print(f"Before preprocessing - Data shape: {data.shape}")
print(f"Before preprocessing - Missing values:\n{data.isnull().sum()}\n")

data = drop_sparse_columns(data, threshold=0.5)
print(f"After drop_sparse - Data shape: {data.shape}\n")

data = fill_missing(data)
print(f"After fill_missing - Data shape: {data.shape}")
print(f"After fill_missing - Any NaN left?\n{data.isnull().sum()}\n")

data = add_year_column(data)
y = data[target]
X = data.drop(columns=[target])

X_train = X[X.index <= 2010]
X_test = X[X.index > 2010]
y_train = y[y.index <= 2010]
y_test = y[y.index > 2010]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"\nX_train columns: {X_train.columns.tolist()}")
print(f"\ny_train values:\n{y_train}")
print(f"\ny_test values:\n{y_test}")
print(f"\nX_train sample:\n{X_train.head()}")
