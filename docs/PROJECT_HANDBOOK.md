# Tunisia Water Stress ML - Project Handbook

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technical Stack](#technical-stack)
3. [Development Setup](#development-setup)
4. [Data Pipeline](#data-pipeline)
5. [Model Training](#model-training)
6. [API Deployment](#api-deployment)
7. [Testing & Quality](#testing--quality)
8. [Code Organization](#code-organization)
9. [Common Tasks](#common-tasks)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

**Tunisia Water Stress ML** is a machine learning system for predicting water stress in Tunisia using World Bank environmental indicators. 

- **Target Variable**: Freshwater withdrawals as % of renewable resources
- **Data Source**: World Bank Open Data API
- **Primary Use Case**: Policy planning and water resource management decisions
- **Model Scope**: Time-series prediction with 1960-2010 training period, 2011-2024 test period

### Key Features

- ✓ Time-aware temporal train/test split (prevents data leakage)
- ✓ Automated feature engineering (lags, temporal features)
- ✓ Multicollinearity detection and handling
- ✓ Multiple baseline models (Linear Regression, Decision Tree, Random Forest, Ridge, Lasso)
- ✓ Comprehensive evaluation metrics (MAE, RMSE, R², cross-validation)
- ✓ FastAPI REST endpoint for real-time predictions
- ✓ 52+ unit and integration tests (100% coverage on critical modules)
- ✓ Full type hints and pre-commit hooks

---

## Technical Stack

### Core ML & Data Processing
- **Python 3.9+** - Programming language
- **scikit-learn 1.0+** - ML models and metrics
- **pandas 1.3+** - Data manipulation
- **numpy 1.20+** - Numerical computing
- **matplotlib 3.3+** - Visualization
- **seaborn 0.11+** - Statistical plots

### API & Deployment
- **FastAPI 0.95+** - REST API framework
- **Uvicorn 0.21+** - ASGI server
- **Pydantic 2.0+** - Data validation
- **Docker** - Containerization

### Development & Testing
- **pytest 8.0+** - Unit testing framework
- **pytest-cov 4.1+** - Coverage reporting
- **pytest-asyncio 0.23+** - Async test support
- **black 24.1+** - Code formatting
- **flake8 7.0+** - Linting
- **mypy 1.8+** - Type checking
- **pre-commit 3.6+** - Git hooks automation
- **isort 5.13+** - Import sorting
- **bandit 1.7+** - Security scanning

### Documentation
- **Sphinx 7.0+** - Documentation generator
- **sphinx-rtd-theme** - ReadTheDocs Sphinx theme

---

## Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/FarahR01/tunisia_water_stress.git
cd tunisia_water_stress_ml
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
& venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Production dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r api_requirements.txt

# Install pre-commit hooks
pre-commit install
```

### 4. Verify Installation

```bash
# Run test suite
python -m pytest tests/ -v --cov=src/ --cov-report=html

# Check code quality
python -m black src/ tests/ api/ --line-length=100
python -m flake8 src/ tests/ api/
python -m mypy src/ --ignore-missing-imports
```

---

## Data Pipeline

### Data Flow

```
World Bank API
     ↓
data/raw/environment_tun.csv
     ↓
src/data_loader.py (load_and_pivot)
     ↓
data/processed/processed_tunisia.csv (wide format)
     ↓
src/preprocessing.py (clean & select features)
     ↓
src/feature_engineering.py (create lags, add year)
     ↓
Training/Test Split (temporal: 1960-2010 → 2011-2024)
     ↓
src/train.py (train models)
     ↓
models/ (saved models, metrics, plots)
```

### Key Processing Steps

#### 1. Data Loading (`src/data_loader.py`)

**Function**: `load_and_pivot(raw_csv_path: str, processed_path: Optional[str] = None) -> pd.DataFrame`

- Loads CSV from World Bank format (long format: years, indicators)
- Filters for Tunisia only
- Pivots to wide format (years as index, indicators as columns)
- Handles missing years with forward/backward fill

**Example:**
```python
from src.data_loader import load_and_pivot

df = load_and_pivot('data/raw/environment_tun.csv', 'data/processed/processed_tunisia.csv')
print(df.shape)  # (years, indicators)
```

#### 2. Preprocessing (`src/preprocessing.py`)

**Functions:**
- `drop_sparse_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame`
  - Removes columns with >50% missing values
- `fill_missing(df: pd.DataFrame) -> pd.DataFrame`
  - Interpolates missing values using forward/backward fill
- `select_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame`
  - Selects specific feature columns

#### 3. Feature Engineering (`src/feature_engineering.py`)

**Functions:**
- `add_lag_features(df: pd.DataFrame, columns: List[str], lags: int = 1) -> pd.DataFrame`
  - Creates lagged features (t-1, t-2, etc.)
- `add_year_column(df: pd.DataFrame) -> pd.DataFrame`
  - Adds year as numeric feature

#### 4. Train/Test Split (Temporal)

**Critical**: Uses temporal split, NOT random split
- Training: 1960-2010 (51 years)
- Testing: 2011-2024 (14 years)
- Prevents data leakage from future information

---

## Model Training

### Training Script

**Command:**
```bash
python src/train.py --processed data/processed/processed_tunisia.csv --models_dir models/ --tune_hyperparams --collinearity_filter --leakage_filter
```

**Options:**
```
--processed: Path to processed CSV file
--models_dir: Directory to save trained models
--train_end: Last year for training split (default: 2010)
--lags: Number of lag features to create (default: 1)
--tune_hyperparams: Enable hyperparameter tuning (GridSearchCV)
--collinearity_filter: Drop highly correlated features (|corr| >= 0.95)
--leakage_filter: Drop features with |corr| >= 0.99 with target
--use_ridge: Train Ridge regression (L2 regularization)
--use_lasso: Train Lasso regression (L1 regularization)
```

### Models

| Model | Type | When to Use | Hyperparameters |
|-------|------|-------------|-----------------|
| **Linear Regression** | Baseline | Simple relationships | None (closed-form) |
| **Ridge** | Regularized Linear | Multicollinearity | alpha (L2 penalty) |
| **Lasso** | Sparse Linear | Feature selection | alpha (L1 penalty) |
| **Decision Tree** | Tree | Interpretability | max_depth, min_samples_split |
| **Random Forest** | Ensemble | Best performance | n_estimators, max_depth |
| **XGBoost** | Boosting | High accuracy (optional) | learning_rate, max_depth, n_estimators |

### Evaluation Metrics

Saved in `models/metrics.csv`:

- **MAE** (Mean Absolute Error): Average prediction error (same units as target)
- **RMSE** (Root Mean Square Error): Penalizes large errors
- **R²** (Coefficient of Determination): Proportion of variance explained (0-1)
- **Train/Test Split**: Ensures honest evaluation

---

## API Deployment

### Running the API

**Local Development:**
```bash
python api/main.py
# or
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**With Docker:**
```bash
docker build -t tunisia-water-stress-api .
docker run -p 8000:8000 tunisia-water-stress-api
```

### API Endpoints

#### GET /health
Health check endpoint.

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok", "version": "1.0.0"}
```

#### POST /v1/predict
Predict water stress for given indicator values.

**Request:**
```json
{
  "features": {
    "indicator_1": 23.5,
    "indicator_2": 45.2,
    "year": 2023
  },
  "model_name": "RandomForest"
}
```

**Response:**
```json
{
  "prediction": 34.7,
  "model_name": "RandomForest",
  "confidence": 0.92,
  "timestamp": "2024-02-22T10:30:00Z"
}
```

#### POST /v1/batch-predict
Predict for multiple samples.

**Request:**
```json
{
  "data": [
    {"indicator_1": 23.5, "indicator_2": 45.2, "year": 2023},
    {"indicator_1": 24.1, "indicator_2": 46.0, "year": 2024}
  ],
  "model_name": "RandomForest"
}
```

**Response:**
```json
{
  "predictions": [34.7, 35.2],
  "model_name": "RandomForest",
  "processing_time_ms": 45
}
```

### API Configuration

Edit `api/config.py`:
```python
# Model selection
DEFAULT_MODEL = "RandomForest"
MODEL_PATHS = {
    "RandomForest": "models_tuned/RandomForest.joblib",
    "Ridge": "models_tuned/Ridge.joblib",
}

# Feature names
FEATURE_NAMES = ["indicator1", "indicator2", ...]

# API settings
API_VERSION = "1.0.0"
MAX_BATCH_SIZE = 1000
```

---

## Testing & Quality

### Run Tests

```bash
# All tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src/ --cov-report=html

# Specific test file
python -m pytest tests/test_preprocessing.py -v

# Run specific test
python -m pytest tests/test_preprocessing.py::TestFillMissing::test_fill_missing_forward_fill -v
```

### Code Quality Checks

```bash
# Format code
python -m black src/ tests/ api/ --line-length=100

# Lint
python -m flake8 src/ tests/ api/

# Type check
python -m mypy src/ --ignore-missing-imports

# Sort imports
python -m isort src/ tests/ api/ --profile black

# Security scan
python -m bandit -r src/ api/ -f json
```

### Pre-commit Hooks

Hooks run automatically before each commit. Configure in `.pre-commit-config.yaml`:

- **black** - Code formatting
- **flake8** - Linting
- **mypy** - Type checking
- **isort** - Import sorting
- **bandit** - Security checks
- **trailing-whitespace** - Remove trailing spaces
- **end-of-file-fixer** - Ensure newline at EOF
- **check-yaml** - Validate YAML syntax
- **check-merge-conflict** - Detect merge conflicts

---

## Code Organization

```
tunisia_water_stress_ml/
├── docs/                          # Documentation
│   ├── PROJECT_HANDBOOK.md        # This file
│   ├── ARCHITECTURE.md            # System architecture
│   ├── DECISIONS.md               # Design decisions
│   └── API.md                     # API documentation
├── src/                           # Source code (ML pipeline)
│   ├── data_loader.py             # Load & pivot data
│   ├── preprocessing.py           # Clean & select features
│   ├── feature_engineering.py     # Create features
│   ├── train.py                   # Training orchestration
│   ├── evaluate.py                # Metrics & plots
│   ├── predict_future.py          # Make predictions
│   └── ...
├── api/                           # FastAPI application
│   ├── main.py                    # API entry point
│   ├── config.py                  # Configuration
│   ├── schemas.py                 # Request/response models
│   ├── model_service.py           # Model loading & inference
│   ├── routers/
│   │   ├── v1.py                  # v1 endpoints
│   │   └── __init__.py
│   └── ...
├── tests/                         # Test suite
│   ├── test_data_loader.py        # Data loading tests
│   ├── test_preprocessing.py      # Preprocessing tests
│   ├── test_feature_engineering.py# Feature eng tests
│   ├── test_pipeline_integration.py# End-to-end tests
│   ├── test_api.py                # API endpoint tests
│   ├── test_model_service.py      # Model service tests
│   └── conftest.py                # Pytest fixtures
├── data/
│   ├── raw/                       # World Bank CSVs
│   └── processed/                 # Cleaned, wide-format data
├── models/                        # Trained models & results
│   ├── RandomForest.joblib        # Saved model
│   ├── metrics.csv                # Performance metrics
│   └── *.png                      # Evaluation plots
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_inspection.ipynb
│   └── 03_modeling.ipynb
├── README.md                      # Project overview
├── CONTRIBUTING.md               # Contribution guidelines
├── requirements.txt              # Production dependencies
├── api_requirements.txt          # Dev/API dependencies
├── .pre-commit-config.yaml       # Pre-commit hooks config
├── .gitignore                    # Git ignore patterns
├── docker-compose.yml            # Multi-container setup
├── Dockerfile                    # API container image
└── nginx.conf                    # Nginx reverse proxy config
```

---

## Common Tasks

### Add a New Feature

1. **Create feature in `src/feature_engineering.py`:**
```python
def calculate_stress_index(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate custom stress index from indicators."""
    df['stress_index'] = df['withdrawal'] / df['renewable']
    return df
```

2. **Add tests in `tests/test_feature_engineering.py`:**
```python
def test_calculate_stress_index():
    df = pd.DataFrame({'withdrawal': [10, 20], 'renewable': [100, 100]})
    result = calculate_stress_index(df)
    assert 'stress_index' in result.columns
    assert result['stress_index'].iloc[0] == 0.1
```

3. **Format & test:**
```bash
python -m black src/feature_engineering.py
python -m pytest tests/test_feature_engineering.py -v
```

4. **Commit:**
```bash
git add src/feature_engineering.py tests/test_feature_engineering.py
git commit -m "feat(features): add stress index calculation"
```

### Train Models with New Data

```bash
# Download latest world bank data (if needed)
python scripts/download_data.py --output data/raw/environment_tun.csv

# Process data
python src/data_loader.py

# Retrain models
python src/train.py --processed data/processed/processed_tunisia.csv --models_dir models/ --tune_hyperparams

# Evaluate
python src/evaluate.py models/metrics.csv
```

### Update API with New Model

1. **Save new model:**
```python
# In src/train.py
import joblib
joblib.dump(model, 'models/NewModel.joblib')
```

2. **Update API config (`api/config.py`):**
```python
MODEL_PATHS = {
    "RandomForest": "models_tuned/RandomForest.joblib",
    "NewModel": "models_tuned/NewModel.joblib",  # Add this
}
```

3. **Restart API:**
```bash
docker-compose restart api
```

### Deploy to Production

```bash
# 1. Merge to main: Create PR, get approval, merge
git checkout main
git pull origin main

# 2. Deploy: Push to main, triggers CI/CD
git push origin main

# 3. Available at: https://tunisia-water-stress-api.herokuapp.com/
```

---

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'sklearn'`

**Solution:**
```bash
pip install scikit-learn
# or reinstall all deps
pip install -r requirements.txt
```

### Data Not Found

**Problem:** `FileNotFoundError: data/raw/environment_tun.csv`

**Solution:**
```bash
# Download data
python scripts/download_data.py --output data/raw/environment_tun.csv

# Or manually:
# Visit: https://data.worldbank.org/indicator/ER.H2O.FWST.UR.ZS
# Download Tunisia data as CSV
# Place in data/raw/environment_tun.csv
```

### Tests Failing

**Problem:** `FAILED tests/test_preprocessing.py::TestFillMissing`

**Solution:**
```bash
# Increase verbosity
python -m pytest tests/test_preprocessing.py -vv -s

# Check data fixtures
python -c "from tests.conftest import *; print(sample_df)"

# Run with debugging
python -m pytest tests/test_preprocessing.py --pdb
```

### API Won't Start

**Problem:** `Address already in use: ('0.0.0.0', 8000)`

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill process and restart
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# Or use different port
uvicorn api.main:app --port 8001
```

### Type Checking Errors

**Problem:** `error: Argument 1 to "load_and_pivot" has incompatible type`

**Solution:**
```bash
# Check mypy errors
python -m mypy src/ --ignore-missing-imports

# Fix type hints in code
# Example: def load_and_pivot(raw_csv_path: str) -> pd.DataFrame:
```

---

## Further Reading

- [Architecture Diagram](./ARCHITECTURE.md) - System design and data flow
- [Design Decisions](./DECISIONS.md) - Why we chose Ridge over XGBoost, etc.
- [API Documentation](./API.md) - Detailed API endpoint reference
- [Contributing Guidelines](../CONTRIBUTING.md) - How to contribute
- [README](../README.md) - Project overview

---

**Last Updated:** February 2026  
**Maintained by:** Project Contributors
