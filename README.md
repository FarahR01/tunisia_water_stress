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

Repository structure

```
tunisia_water_stress_ml/
├── docs/                          # 📚 Project documentation
│   ├── PROJECT_HANDBOOK.md        # Comprehensive development guide
│   ├── ARCHITECTURE.md            # System design + Mermaid diagrams
│   ├── DECISIONS.md               # Design decisions & trade-offs
│   └── API.md                     # API endpoint reference
│
├── src/                           # 🧠 ML Pipeline (core logic)
│   ├── data_loader.py             # Load World Bank CSVs → wide format
│   ├── preprocessing.py           # Clean, fill missing, select features
│   ├── feature_engineering.py     # Create lags, temporal features
│   ├── train.py                   # Orchestrate full pipeline
│   ├── evaluate.py                # Metrics & visualization
│   ├── predict_future.py          # Generate forecasts
│   ├── hyperparameter_tuning.py   # GridSearchCV automation
│   └── feature_importance.py      # Feature analysis
│
├── api/                           # 🚀 FastAPI REST Service
│   ├── main.py                    # App entry point
│   ├── config.py                  # Configuration
│   ├── schemas.py                 # Request/response models
│   ├── model_service.py           # Model loading & inference
│   ├── dependencies.py            # Dependency injection
│   ├── logging_config.py          # Logging setup
│   └── routers/
│       ├── v1.py                  # v1 endpoints
│       └── __init__.py
│
├── tests/                         # ✅ Test Suite (52 tests)
│   ├── test_data_loader.py        # Data loading tests (9)
│   ├── test_preprocessing.py      # Preprocessing tests (16)
│   ├── test_feature_engineering.py# Feature engineering tests (18)
│   ├── test_pipeline_integration.py# End-to-end tests (8)
│   ├── test_api.py                # API endpoint tests
│   ├── test_model_service.py      # Model service tests
│   ├── conftest.py                # Shared fixtures
│   └── __pycache__/
│
├── data/
│   ├── raw/                       # World Bank indicator CSVs
│   │   └── environment_tun.csv    # Tunisia environment data (long format)
│   └── processed/
│       └── processed_tunisia.csv  # Cleaned, wide-format (ready for ML)
│
├── models/                        # 📊 Trained Models & Results
│   ├── RandomForest.joblib        # Trained model
│   ├── Ridge.joblib               # Regularized linear model
│   ├── metrics.csv                # Performance metrics (MAE, RMSE, R²)
│   ├── *_actual_vs_pred.png       # Prediction plots
│   └── *_feature_importance.png   # Feature importance plots
│
├── notebooks/                     # 📓 Jupyter Notebooks
│   ├── 01_data_exploration.ipynb  # EDA, data quality checks
│   ├── 02_model_inspection.ipynb  # Model plots & correlation analysis
│   └── 03_modeling.ipynb          # Full training walkthrough
│
├── .pre-commit-config.yaml        # Pre-commit hooks (black, flake8, mypy, bandit)
├── .gitignore                     # Git ignore patterns
├── README.md                      # Project overview (you are here)
├── CONTRIBUTING.md                # Contribution guidelines
├── requirements.txt               # Production dependencies
├── api_requirements.txt           # Dev + API dependencies
│
├── CODE_QUALITY_SUMMARY.md        # Code quality implementation details
├── IMPLEMENTATION_STATUS.md       # Feature completion checklist
├── FINAL_REPORT.md                # Project completion report
│
├── docker-compose.yml             # Multi-container orchestration
├── Dockerfile                     # API container image
├── nginx.conf                     # Reverse proxy configuration
│
└── scripts/
    ├── check_correlations.py      # Correlation analysis utility
    └── predict_2030.py            # Future predictions script
```

---

## Architecture Overview

**Data Flow:**
```
World Bank API (Open Data)
         ↓
   data/raw/*.csv
         ↓ (load_and_pivot)
   DataFrame (wide format)
         ↓ (preprocessing)
   Clean & selected features
         ↓ (feature_engineering)
   Lagged & temporal features
         ↓ (temporal split)
   Separate train (1960-2010) and test (2011-2024) sets
         ↓ (train.py)
   Trained models (.joblib) + metrics.csv
         ↓
   ┌─────────────────────────────────────┐
   │ Option 1: Batch (notebooks)         │
   │ Option 2: API (/v1/predict)         │
   │ Option 3: CLI (predict_future.py)   │
   └─────────────────────────────────────┘
```

**Why Temporal Split?** 
- ✓ Prevents data leakage (no future info in training)
- ✓ Realistic evaluation (how model performs on unseen future)
- ✗ (Incorrect) Random split would artificially inflate accuracy

## Quick Start

### Option 1: Run Training Pipeline

**Step 1: Set up environment**
```powershell
# Create virtual environment
python -m venv venv

# Activate (Windows)
& venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Step 2: Train models**
```powershell
# Basic training
python src/train.py

# With all options (recommended)
python src/train.py --models_dir models/ --tune_hyperparams --leakage_filter --collinearity_filter
```

**Step 3: Review results**
- Models saved to `models/` (`.joblib` files)
- Metrics in `models/metrics.csv`
- Plots in `models/` (`.png` files)
- Open `notebooks/02_model_inspection.ipynb` for visualization

### Option 2: Use REST API

**Start the API:**
```powershell
pip install -r api_requirements.txt
uvicorn api.main:app --reload
```

Navigate to `http://localhost:8000/docs` for interactive API documentation.

**Example predictions:**
```bash
curl -X POST "http://localhost:8000/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {"indicator_1": 23.5, "year": 2023}, "model_name": "RandomForest"}'
```

### Option 3: Use Docker

```bash
# Build and run API container
docker-compose up -d

# API available at: http://localhost:8000
```

---

## Documentation

Complete guides available in `docs/`:

- **[PROJECT_HANDBOOK.md](docs/PROJECT_HANDBOOK.md)** ← Start here for detailed development guide
  - Development setup and environment configuration
  - Data pipeline walk-through with code examples
  - Model training options and results interpretation
  - API deployment and usage
  - Common tasks and troubleshooting

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design and data flow
  - Data pipeline diagram (Mermaid)
  - System architecture diagram
  - Component interaction flow
  - Technology stack by layer
  - Deployment architecture
  - Scalability considerations

- **[DECISIONS.md](docs/DECISIONS.md)** - Why we chose specific technologies
  - Why Ridge/Lasso over XGBoost (with data size considerations)
  - Why temporal train/test split (prevents data leakage)
  - Multicollinearity and leakage detection strategies
  - Architecture choices (FastAPI, Docker, modular design)
  - Testing and quality assurance decisions

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute
  - Git workflow and commit message format
  - Branch naming conventions
  - Type hints and code quality requirements
  - Testing requirements (100% coverage on src/)
  - Pre-commit hooks setup

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

