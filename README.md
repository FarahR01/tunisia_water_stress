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
├── tests/                         # ✅ Test Suite
│   ├── test_data_loader.py        # Data loading tests
│   ├── test_preprocessing.py      # Preprocessing tests
│   ├── test_feature_engineering.py# Feature engineering tests
│   ├── test_pipeline_integration.py# End-to-end tests
│   ├── test_api.py                # API endpoint tests
│   ├── test_model_service.py      # Model service tests
│   ├── test_schemas.py            # Schema validation tests
│   ├── conftest.py                # Shared fixtures
│   └── __pycache__/
│
├── data/
│   ├── raw/                       # World Bank indicator CSVs
│   ├── processed/                 # Cleaned, processed data
│   └── DATA_VERSION.md            # Data versioning info
│
├── models/                        # 📊 Trained Models & Results
│   ├── RandomForest.joblib        # Trained model
│   ├── Ridge.joblib               # Regularized linear model
│   ├── DecisionTree.joblib        # Decision tree model
│   ├── metrics.csv                # Performance metrics (MAE, RMSE, R²)
│   └── feature_importance_summary.csv
│
├── models_tuned/                  # Hyperparameter-tuned models
│   ├── RandomForest.joblib        # Tuned model
│   ├── Ridge.joblib               # Tuned model
│   ├── Lasso.joblib               # Tuned model
│   ├── DecisionTree.joblib        # Tuned model
│   ├── metrics.csv                # Tuned model metrics
│   ├── hyperparameter_tuning_summary.csv
│   ├── *_cv_results.csv           # Cross-validation results
│   └── collinearity_dropped.txt   # Dropped features due to collinearity
│
├── artifacts/
│   ├── models_tuned/              # Additional tuned model artifacts
│   └── predictions/               # Future predictions & scenario analysis
│
├── config/
│   └── train_config.yaml          # Training configuration
│
├── dashboard/                     # 📊 Streamlit Dashboard
│   ├── app.py                     # Dashboard main app
│   ├── Dockerfile                 # Dashboard container
│   ├── docker-compose.yml         # Dashboard orchestration
│   ├── requirements.txt           # Dashboard dependencies
│   ├── components/                # Dashboard UI components
│   ├── pages/                     # Dashboard pages
│   └── utils/                     # Dashboard utilities
│
├── notebooks/                     # 📓 Jupyter Notebooks
│   ├── 01_data_exploration.ipynb  # EDA, data quality checks
│   ├── 02_model_inspection.ipynb  # Model plots & correlation analysis
│   └── 03_modeling.ipynb          # Full training walkthrough
│
├── scripts/
│   ├── check_correlations.py      # Correlation analysis utility
│   └── predict_2030.py            # Future predictions script
│
├── .pre-commit-config.yaml        # Pre-commit hooks (black, flake8, mypy, bandit)
├── .gitignore                     # Git ignore patterns
├── README.md                      # Project overview (you are here)
├── LICENSE                        # MIT License
├── CONTRIBUTING.md                # Contribution guidelines
├── requirements.txt               # Production dependencies
├── api_requirements.txt           # API dependencies
├── docker-compose.yml             # Multi-container orchestration
├── Dockerfile                     # API container image
├── nginx.conf                     # Reverse proxy configuration
├── run_api.bat                    # Windows API startup script
├── run_api.sh                     # Unix API startup script
│
└── docs/
    └── (See above)
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
