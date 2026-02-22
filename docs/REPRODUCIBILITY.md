# Reproducibility & Experiment Tracking Guide

This document explains how to use the reproducibility and experiment tracking infrastructure in the Tunisia Water Stress ML project.

## Table of Contents

1. [Overview](#overview)
2. [Configuration Management](#configuration-management)
3. [Random Seed Management](#random-seed-management)
4. [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
5. [Data Versioning](#data-versioning)
6. [Running Reproducible Experiments](#running-reproducible-experiments)
7. [Comparing Experiment Results](#comparing-experiment-results)
8. [Best Practices](#best-practices)

---

## Overview

Reproducibility is critical for ML research. This project implements three key components:

| Component | Purpose | Location |
|-----------|---------|----------|
| **Configuration Management** | YAML-based experiment config with Pydantic validation | [config/](../../config/) |
| **Experiment Tracking** | MLflow for logging metrics, parameters, models, and artifacts | [mlruns/](../../mlruns/) |
| **Data Versioning** | SHA256 hashes and metadata for all data assets | [data/DATA_VERSION.md](../../data/DATA_VERSION.md) |
| **Seed Management** | Centralized random seeds for all libraries | [src/config_train.py](../../src/config_train.py) |

---

## Configuration Management

Instead of using CLI flags, all experiment parameters are defined in YAML config files. This makes experiments self-documenting and easy to reproduce.

### Create a New Experiment Config

```bash
# Copy the example config
cp config/train_config.yaml config/my_experiment.yaml

# Edit the config with your parameters
# (see detailed options below)
```

### Config File Structure

```yaml
# Reproducibility seeds
seeds:
  numpy_seed: 42
  python_seed: 42
  sklearn_seed: 42
  xgboost_seed: 42

# Data settings
data:
  raw_data_path: "data/raw/environment_tun.csv"
  processed_data_path: "data/processed/processed_tunisia.csv"
  sparse_threshold: 0.5
  explicit_features: null                    # Leave null to auto-select
  target_column: null                        # Leave null to auto-detect

# Feature engineering
feature_engineering:
  lag_features: 0
  apply_leakage_filter: true
  leakage_correlation_threshold: 0.99
  apply_collinearity_filter: true
  collinearity_threshold: 0.95
  max_features: 12

# Train/test split
split:
  train_end_year: 2010

# Models to train
models:
  enable_hyperparameter_tuning: false
  linear_regression:
    enabled: true
    hyperparameters: {}
  random_forest:
    enabled: true
    hyperparameters:
      n_estimators: 100
      random_state: 42
  xgboost:
    enabled: false
    hyperparameters: {}

# MLflow experiment tracking
mlflow:
  enabled: true
  tracking_uri: "./mlruns"
  experiment_name: "water_stress_regression"
  log_models: true
  log_plots: true

# Output directory
output_dir: "models_experiment_001"
```

### Load Config Programmatically

```python
from src.config_train import TrainConfig

# Load from YAML file
config = TrainConfig.from_yaml("config/my_experiment.yaml")

# Or create programmatically with validation
from src.config_train import TrainConfig, SeedConfig, DataConfig

config = TrainConfig(
    seeds=SeedConfig(numpy_seed=123),
    data=DataConfig(max_features=15),
    output_dir="models_custom"
)

# Save config
config.to_yaml("config/saved_experiment.yaml")

# Access values
print(config.seeds.numpy_seed)          # 123
print(config.data.sparse_threshold)     # 0.5
print(config.models.random_forest.enabled)  # True
```

---

## Random Seed Management

All random seeds are centralized in `config.seeds` for complete reproducibility.

### Seeds Configuration

```python
from src.config_train import SeedConfig

seeds = SeedConfig(
    numpy_seed=42,      # NumPy random operations
    python_seed=42,     # Python's random module
    sklearn_seed=42,    # Scikit-learn models (DecisionTree, RandomForest, Ridge, Lasso)
    xgboost_seed=42     # XGBoost gradient boosting
)
```

### Using Seeds in Your Code

```python
import random
import numpy as np
from src.config_train import TrainConfig

config = TrainConfig.from_yaml("config/train_config.yaml")

# Set seeds at the start of your script
np.random.seed(config.seeds.numpy_seed)
random.seed(config.seeds.python_seed)

# All subsequent operations use these seeds
# Models are created with sklearn_seed and xgboost_seed
```

### Guaranteed Reproducibility

With the same config file and seeds:
1. Data preprocessing produces identical results
2. Train/test splits are identical
3. Model initialization and training is deterministic
4. Cross-validation folds are identical

**Example:** Running the same experiment twice produces identical metrics to machine precision.

---

## Experiment Tracking with MLflow

MLflow tracks all experiment runs, making it easy to compare results and understand what changed between attempts.

### What Gets Logged

For each experiment run, MLflow logs:

- **Parameters:** All config settings (seeds, model hyperparameters, feature settings)
- **Metrics:** Model performance (R², MAE, RMSE, etc.) for train and test sets
- **Models:** Trained model files (joblib format)
- **Artifacts:** Plots (feature importance, predictions), metrics CSV, config YAML
- **Tags:** Experiment metadata (framework version, run date, etc.)

### MLflow Tracking Locations

```
mlruns/
├── 0/                                  # Experiment ID (auto-assigned)
│   ├── meta.yaml
│   └── runs/
│       ├── abc123def456.../            # Run 1
│       │   ├── params/                 # Parameters logged
│       │   ├── metrics/                # Metrics (one file per step)
│       │   ├── artifacts/
│       │   │   ├── models/
│       │   │   ├── plots/
│       │   │   └── config.yaml
│       │   └── meta.yaml
│       └── xyz789uvw012.../            # Run 2
└── 1/                                  # Another experiment
```

### View MLflow UI

```bash
# Start local MLflow tracking server
mlflow ui --backend-store-uri ./mlruns

# Open in browser: http://localhost:5000
```

The UI shows:
- All experiments and runs
- Parameter and metric comparisons
- Artifact visualization
- Model registry

### Using MLflow in Your Code

```python
from src.mlflow_utils import MLflowTracker
from src.config_train import TrainConfig

config = TrainConfig.from_yaml("config/train_config.yaml")

# Create tracker
tracker = MLflowTracker.from_config(config.mlflow)

# Track an experiment
with tracker.track_experiment(config=config) as logger:
    # Your training code
    model = train_model(...)
    metrics = evaluate_model(model)

    # Log results (automatic when using context manager)
    logger.log_metrics(metrics)
    logger.log_model(model, "artifacts/model")

    # Get run ID for future reference
    run_id = logger.get_run_id()
    print(f"Tracked experiment: {run_id}")
```

### Programmatic Result Analysis

```python
from src.mlflow_utils import ExperimentAnalyzer

analyzer = ExperimentAnalyzer(tracking_uri="./mlruns")

# Get all runs for an experiment
runs = analyzer.get_experiment_runs("water_stress_regression")

# Find best run based on metric
best_run = analyzer.get_best_run(runs, metric_name="test_r2", mode="max")
print(f"Best run: {best_run.run_id}")
print(f"R² Score: {best_run.metrics.get('test_r2')}")
```

---

## Data Versioning

Track data versions to ensure experiment reproducibility across time and environments.

### Data Manifest

```python
from src.data_versioning import DataManifest

# Create manifest
manifest = DataManifest()
manifest.add_file(
    "data/raw/environment_tun.csv",
    source="World Bank API",
    description="Water/environment indicators for Tunisia (1972-2020)"
)
manifest.add_file(
    "data/processed/processed_tunisia.csv",
    source="data_loader.py pipeline",
    description="Processed wide-form dataset"
)

# Save manifest
manifest.save("data/DATA_MANIFEST.json")
```

### Verify Data Integrity

```python
from src.data_versioning import DataManifest

# Load saved manifest
manifest = DataManifest.load("data/DATA_MANIFEST.json")

# Verify all files
results = manifest.verify_all()
for filepath, is_valid in results.items():
    status = "✓" if is_valid else "✗ CHANGED"
    print(f"{status} {filepath}")

# Check specific file
if manifest.verify_file("data/raw/environment_tun.csv"):
    print("Data is unchanged since last run")
else:
    print("WARNING: Data has changed!")
    print(f"Expected hash: {manifest.get_hash('data/raw/environment_tun.csv')}")
    print(f"Current hash: {compute_file_hash('data/raw/environment_tun.csv')}")
```

### Data Documentation

See [data/DATA_VERSION.md](../../data/DATA_VERSION.md) for:
- Data source information
- Last update date
- File hashes and sizes
- Data transformation pipeline details

---

## Running Reproducible Experiments

### Step 1: Create Config

```bash
# Copy and customize config
cp config/train_config.yaml config/experiment_v1.yaml
# Edit config with your parameters
```

### Step 2: Update Requirements (if needed)

```bash
# Install/update dependencies
pip install -r requirements.txt
```

### Step 3: Run Training

```bash
# Using new config-based system (coming soon with updated train.py)
python src/train.py --config config/experiment_v1.yaml

# The training script will:
# 1. Load config and set all random seeds
# 2. Log all parameters to MLflow
# 3. Train models with reproducible results
# 4. Save models and metrics to output_dir
# 5. Log artifacts to MLflow
```

### Step 4: View Results

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Or use Python
from src.mlflow_utils import ExperimentAnalyzer
analyzer = ExperimentAnalyzer("./mlruns")
runs = analyzer.get_experiment_runs("water_stress_regression")
# Compare results...
```

---

## Comparing Experiment Results

### Compare Runs in MLflow UI

1. Navigate to MLflow UI (http://localhost:5000)
2. Select experiment "water_stress_regression"
3. View runs side-by-side
4. Compare metrics, parameters, artifacts

### Programmatic Comparison

```python
import pandas as pd
from src.mlflow_utils import ExperimentAnalyzer

analyzer = ExperimentAnalyzer("./mlruns")
runs = analyzer.get_experiment_runs("water_stress_regression")

# Convert to DataFrame for analysis
runs_df = runs[['params.models.random_forest.hyperparameters.n_estimators',
                 'metrics.test_r2', 'metrics.test_rmse', 'start_time']]

# Find best performing run
best_idx = runs_df['metrics.test_r2'].idxmax()
print(runs_df.loc[best_idx])
```

---

## Best Practices

### 1. **Version Your Configs**

```bash
# Name configs with version/date
config/
├── train_config.yaml              # Template
├── experiment_v1_baseline.yaml
├── experiment_v2_more_features.yaml
├── experiment_v3_tuned_rf.yaml
└── experiment_final_2024_02_22.yaml
```

### 2. **Document Experiment Intent**

```yaml
# In config file:
experiment_notes: |
  Baseline with default hyperparameters
  - Testing data quality with all features
  - Comparing 5 different model types
  - No hyperparameter tuning
```

### 3. **Use Meaningful Experiment Names**

```python
tracker = MLflowTracker(
    experiment_name="water_stress_regression_v2",  # Good: descriptive
    # NOT: "experiment1" or "test"
)
```

### 4. **Save Configs with Results**

```python
# Config is automatically logged as artifact
# But also save locally:
config.to_yaml(f"{output_dir}/config_used.yaml")
```

### 5. **Pin Dependency Versions**

```bash
# Use requirements.txt (already done)
# Ensures same package versions across runs
pip install -r requirements.txt
```

### 6. **Create README for Each Experiment**

```bash
# For each config
cat > config/experiment_v2_README.md << 'EOF'
# Experiment V2: Feature Importance Study

**Date:** 2024-02-22
**Config:** config/experiment_v2_more_features.yaml
**MLflow Experiment:** water_stress_regression

## Changes from V1
- Added product lag features (lag_features: 2)
- Reduced collinearity threshold (0.90 instead of 0.95)
- Enabled hyperparameter tuning

## Expected Results
- Higher R² on test set
- More stable cross-validation scores
EOF
```

### 7. **Always Track Data Versions**

```bash
# Before each major experiment
python -c "
from src.data_versioning import DataManifest
m = DataManifest()
m.add_file('data/raw/environment_tun.csv')
m.add_file('data/processed/processed_tunisia.csv')
m.save('data/DATA_MANIFEST.json')
"
```

### 8. **Use Consistent Seeds**

```python
# Always use 42 or document why different
SeedConfig(
    numpy_seed: 42,
    python_seed: 42,
    sklearn_seed: 42,
    xgboost_seed: 42
)
```

---

## Troubleshooting

### MLflow not tracking runs?

```python
# Check if enabled in config
config.mlflow.enabled  # Should be True

# Verify MLflow installed
pip install mlflow

# Check tracking URI
import mlflow
mlflow.set_tracking_uri("./mlruns")
print(mlflow.get_tracking_uri())
```

### Data hash mismatch?

```python
from src.data_versioning import DataManifest, compute_file_hash

# Check what changed
manifest = DataManifest.load("data/DATA_MANIFEST.json")
expected = manifest.get_hash("data/raw/environment_tun.csv")
current = compute_file_hash("data/raw/environment_tun.csv")

print(f"Expected: {expected}")
print(f"Current: {current}")
print(f"Match: {expected == current}")
```

### Experiment not reproducible?

1. **Check seeds:** Are all numpy_seed, python_seed, sklearn_seed the same?
2. Check **data:** Did data files change? Run data manifest verification.
3. **Check library versions:** Are scikit-learn, numpy, pandas versions identical?
4. **Check config:** Are all model hyperparameters identical?

---

## Summary

| Goal | How |
|------|-----|
| Run reproducible experiment | Use YAML config + MLflow tracking |
| Track metric changes | View MLflow UI or programmatic analysis |
| Compare experiments | Load runs from MLflow and compare |
| Verify data unchanged | Use DataManifest with SHA256 hashes |
| Document experiment | Add notes to config YAML |
| Clone experiment | Copy config file, run training |
| Share experiment | Commit config to git + export MLflow artifacts |

---

## Related Documentation

- [Config File Reference](../../config/train_config.yaml)
- [Data Versioning](../../data/DATA_VERSION.md)
- [MLflow Official Docs](https://mlflow.org/docs/latest/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
