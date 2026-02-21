# Modeling and Predicting Water Stress in Tunisia

Project goal: model and predict the level of water stress in Tunisia using World Bank environmental indicators. This repository demonstrates a full data-science workflow: data ingestion, cleaning, feature selection, time-aware model training, evaluation, and basic interpretation.

**Why this project matters**
- Water stress (freshwater withdrawals as a % of renewable freshwater resources) is a key sustainability indicator for Tunisia.
- Predicting water stress helps in planning, policy evaluation, and prioritizing interventions.

What this repo showcases
- Time-series aware machine learning with a clear temporal train/test split (train: 1960–2010, test: 2011–2024).
- Careful feature selection and preprocessing for World Bank long-format indicator data.
- Baseline models: Linear Regression, Decision Tree, Random Forest, with evaluation (MAE, RMSE, R²).
- Diagnostic steps for multicollinearity and data leakage, with suggested mitigations.

Repository structure

- `data/raw/` — input CSVs (World Bank Tunisia indicators).\
- `data/processed/` — processed wide-form data (years × indicators).\
- `notebooks/01_data_exploration.ipynb` — exploratory analysis.\
- `notebooks/02_model_inspection.ipynb` — displays model plots and correlation diagnostics.\
- `src/data_loader.py` — CSV loader and pivot helper.\
- `src/preprocessing.py` — missing-value handling and selection helpers.\
- `src/feature_engineering.py` — simple feature transforms (lags, year column).\
- `src/train.py` — orchestration script to build data, split temporally, train models, and save results.\
- `src/evaluate.py` — metrics and plotting utilities.\
- `models/` — trained model files, metrics, and plots after running `src/train.py`.

Quick start

1. Create and activate your Python environment, then install requirements:

```powershell
python -m venv venv
& venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the training pipeline (auto-detects target and selects features by keyword):

```powershell
python src\train.py
```

3. Inspect results in `models/` and open `notebooks/02_model_inspection.ipynb` to visualize plots and correlation diagnostics.

Key findings (from an initial run)
- The pipeline auto-detected a water-related target and trained three models. The saved metrics are in `models/metrics.csv` and plots are in `models/`.
- A near-perfect Linear Regression fit was traced to target leakage: the processed features contained indicators that are identical or direct transforms of the target (e.g., "Annual freshwater withdrawals, total (% of internal resources)" and similar). This causes inflated R² and misleading performance.

Recommended next steps
- Remove duplicate/target-leaking indicators before training (drop columns that are identical or have |corr| >= 0.99 with the target).\
- Use regularized linear models (`Ridge`, `Lasso`) to reduce coefficient instability.\
- Use feature selection (drop highly collinear features / use PCA) and hyperparameter tuning for tree ensembles.\
- Expand evaluation: cross-validate using rolling-origin (time-series CV), and produce explainability plots (SHAP) for Random Forest.

What I learned / Demonstrated skills
- Practical handling of long-form World Bank data and reshaping to wide time-series.\
- Time-aware splitting and the importance of avoiding random splits on time-series data.\
- Detecting data leakage and diagnosing multicollinearity using correlation matrices.\
- Building a small, reproducible training pipeline with clear outputs (models, plots, metrics).

Notes and caveats
- The current pipeline is Stage 1 (baselines and diagnostics). Careful feature curation and model tuning are required before trustable policy recommendations can be made.\
- Some World Bank indicators may be sparse or only partially overlapping in years — the preprocessing step interpolates and forward/back-fills where appropriate; review interpolation choices for your analysis goals.

If you want, I can now:
- automatically drop columns that leak the target and re-run training with `Ridge`, or\
- produce a minimal write-up section suitable for GitHub (results, plots, and interpretation).

Contact
- Repo maintained by the project author. Pull requests and issues welcome.

