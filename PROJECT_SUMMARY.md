# Tunisia Water Stress ML — Project Summary

## Project Objectives

- Build a reproducible machine learning pipeline to estimate and forecast national-level water stress for Tunisia.
- Identify the most influential environmental and socioeconomic features driving water stress.
- Provide robust 2030 scenario projections under different extrapolation assumptions to support policy planning.
- Produce tools for model training, evaluation, feature importance analysis, hyperparameter tuning, and scenario prediction.

## What we built

- A modular Python codebase under `src/` that contains:
  - `src/data_loader.py` — data loading and pivoting utilities
  - `src/preprocessing.py` — cleaning, missing value handling and sparse-column dropping
  - `src/feature_engineering.py` — lag creation and `year` feature injection
  - `src/hyperparameter_tuning.py` — GridSearchCV/RandomizedSearchCV wrapper and param grids
  - `src/feature_importance.py` — extract and summarize feature importances
  - `src/train.py` — main training orchestrator (filtering, tuning, model training)
  - `src/evaluate.py` — metrics, plots and CSV reporting
  - `src/predict_future.py` — scenario generation and 2030 prediction utilities
- CLI scripts in `scripts/`:
  - `scripts/predict_2030.py` — generate 2030 predictions for trained models
- Jupyter notebooks for exploration and model inspection in `notebooks/`.
- Data stored in `data/` with `cleaned_water_stress.csv` and processed data at `data/processed/processed_tunisia.csv`.
- Trained model artifacts are in `models/` (baseline) and `models_tuned/` (tuned models). Local experimental artifacts are archived under `artifacts/` (ignored by git).

## Key Features & Pipeline Behavior

- Target leakage detection and filtering: automatically drops features that are nearly identical to the target (configurable via `--leakage_filter`).
- Pairwise collinearity pruning: removes one feature from very-high-correlation pairs (`--collinearity_filter`).
- Hyperparameter tuning: optional tuning with `--tune_hyperparams` (GridSearch/RandomizedSearch with CV).
- Regularized linear models: support for `Ridge` and `Lasso` via CLI flags `--use_ridge`/`--use_lasso`.
- Optional XGBoost integration (conditional import; flagged by `--use_xgboost`).
- Scenario projection methods: `linear`, `exponential`, `average`, and `last_value` implemented for 2030 feature extrapolation.

## Features used in final models

During the cleaning/filtering steps the final trained models used the following 7 features (post leakage/collinearity filtering):

- `Annual freshwater withdrawals, domestic (% of total freshwater withdrawal)`
- `Annual freshwater withdrawals, industry (% of total freshwater withdrawal)`
- `Annual freshwater withdrawals, total (% of internal resources)`
- `Renewable internal freshwater resources per capita (cubic meters)`
- `Renewable internal freshwater resources, total (billion cubic meters)`
- `Water productivity, total (constant 2015 US$ GDP per cubic meter of total freshwater withdrawal)`
- `year`

These features were chosen by removing leakage-correlated indicators and pruning collinear pairs; they represent demand (withdrawals), availability (renewables) and productivity.

## Model suite & results

Trained models (saved under `models_tuned/`):
- `LinearRegression` — baseline linear model
- `Ridge` — regularized linear model (best performer)
- `Lasso` — sparse linear model
- `DecisionTree` — single-tree regressor
- `RandomForest` — ensemble tree regressor

Key reported metrics (example from the final run):
- Ridge: R² = 0.998698, MAE ≈ 0.0656, RMSE ≈ 0.0722
- Lasso: R² ≈ 0.9320
- DecisionTree / RandomForest: lower generalization (R² negative in some runs) — sensitive to time splits and small dataset

Note: Some tree models show poor R² on test data in specific temporal splits; Ridge gave consistently stable, high R² after leakage filtering.

## 2030 Scenario Predictions

- Script: `scripts/predict_2030.py` (uses `src/predict_future.py` internals).
- Methods: `--scenario_method` accepts `linear`, `exponential`, `average`, `last_value`.
- Output: `predictions/water_stress_2030_predictions.csv` and `predictions/water_stress_2030_detailed.txt` and a markdown analysis `predictions/2030_scenario_analysis.md`.

Representative results (Ridge baseline):
- Baseline (2023): ~98.1% water stress (level of water stress indicator)
- Ridge 2030 (linear): ~75.3% — substantial projected improvement under feature trends
- Scenario spread across models/methods: ~74.7% to ~84.9% depending on method and model

## How to run (reproducible commands)

1. Create & activate venv, install deps:

```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train models (example):

```powershell
python src\train.py --processed data\processed\processed_tunisia.csv --models_dir models_tuned --lags 0 --train_end 2010 --leakage_filter --collinearity_filter --tune_hyperparams --use_ridge --use_lasso
```

3. Generate 2030 predictions (linear):

```powershell
python scripts\predict_2030.py --models_dir models_tuned --processed data\processed\processed_tunisia.csv --last_year 2024 --scenario_method linear
```

4. Inspect outputs:
- Model files: `models_tuned/*.joblib`
- Metrics: `models_tuned/metrics.csv`
- Predictions: `predictions/water_stress_2030_predictions.csv`
- Feature importance results: `models_tuned/feature_importance_summary.csv`

## Project organization & branches

- `src/` — core source modules
- `scripts/` — utilities and CLI entrypoints
- `data/` — raw, cleaned, processed datasets
- `notebooks/` — exploratory analysis
- `models/` & `models_tuned/` — model artifacts
- `predictions/` — generated scenario outputs
- `artifacts/` — local archive for large artifacts (ignored in git)

Notable branches in the repository:
- `feature/advanced-ml-pipeline` — all new modeling features, tuning and prediction scripts
- `clean-workspace` — cleaned repository baseline (no temporary clutter)

## Limitations & caveats

- Dataset size is modest (national-level time series); tree-based models can overfit and show unstable R² on temporal splits.
- Projections rely on historic indicator trends; external shocks (climate events, policy changes) will not be captured.
- XGBoost integration is conditional and may require installing platform-compatible binaries (`pip install xgboost`).

## Next steps (suggested)

- Add unit tests for data preprocessing and training functions.
- Add CI workflow (GitHub Actions) to run linting and basic tests on push/PR.
- Create a lightweight REST API (FastAPI) endpoint for predictions and a Streamlit dashboard for visualization.
- Add dataset versioning and reproducible experiment logging (e.g., MLflow or DVC).

## Contacts & provenance

- Primary author: project workspace (local modifications captured in git history).
- For reproducibility, refer to commit history and the `feature/advanced-ml-pipeline` branch which contains the experimental changes described above.

---

If you want, I can also:
- Add this file to `README.md` as a short overview.
- Create a `docs/` folder and move this into a longer-form project handbook.
- Open a PR from `feature/advanced-ml-pipeline` into `clean-workspace` or `master` with this documentation included.

Which would you prefer next?