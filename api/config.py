"""API Configuration settings."""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Model settings
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models_tuned"
DEFAULT_MODEL_NAME = "Lasso"  # Best performing model based on metrics

# Available models
AVAILABLE_MODELS = ["LinearRegression", "DecisionTree", "RandomForest", "Ridge", "Lasso"]

# Data settings
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_tunisia.csv"

# API settings
API_TITLE = "Tunisia Water Stress ML API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
## Tunisia Water Stress Prediction API

This API provides predictions for water stress levels in Tunisia using machine learning models.

### Features:
- **Health Check**: Verify API status
- **Model Info**: Get information about available models and their performance
- **Predictions**: Make water stress predictions for specific years
- **Scenario Analysis**: Generate 2030 scenario predictions
"""

# Feature columns expected by the trained models
# These are the 5 features used after collinearity filtering
FEATURE_COLUMNS = [
    "year",
    "Water productivity, total (constant 2015 US$ GDP per cubic meter of total freshwater withdrawal)",
    "Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)",
    "Annual freshwater withdrawals, industry (% of total freshwater withdrawal)",
    "Renewable internal freshwater resources, total (billion cubic meters)"
]

# Target variable
TARGET_COLUMN = "Level of water stress: freshwater withdrawal as a proportion of available freshwater resources"
