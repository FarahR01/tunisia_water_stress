"""Training configuration management with Pydantic validation and reproducibility.

This module provides centralized configuration for all training experiments, including:
- Reproducibility settings (random seeds)
- Data paths and preprocessing parameters
- Model selection and hyperparameters
- Experiment tracking settings

Usage:
    from config_train import TrainConfig
    config = TrainConfig.from_yaml("config/train_config.yaml")
    # or
    config = TrainConfig(raw_data_path="data/raw/environment_tun.csv", ...)
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# RANDOM SEED MANAGEMENT - Central location for all reproducibility settings
# ============================================================================
class SeedConfig(BaseModel):
    """Centralized random seed configuration for reproducibility.

    Attributes:
        numpy_seed: NumPy random seed for consistent array operations
        python_seed: Python random module seed
        sklearn_seed: Scikit-learn model seed (used in DecisionTree, RandomForest, etc.)
        xgboost_seed: XGBoost model seed for gradient boosting reproducibility

    All models use these seeds during initialization to ensure deterministic behavior.
    """

    numpy_seed: int = Field(default=42, description="NumPy random seed")
    python_seed: int = Field(default=42, description="Python random module seed")
    sklearn_seed: int = Field(default=42, description="Scikit-learn model seed")
    xgboost_seed: int = Field(default=42, description="XGBoost model seed")

    class Config:
        extra = "forbid"
        validate_assignment = True

    def __str__(self) -> str:
        """String representation of seed configuration."""
        return (
            f"SeedConfig(numpy={self.numpy_seed}, python={self.python_seed}, "
            f"sklearn={self.sklearn_seed}, xgboost={self.xgboost_seed})"
        )


# ============================================================================
# DATA CONFIGURATION
# ============================================================================
class DataConfig(BaseModel):
    """Data preprocessing and feature selection configuration.

    Attributes:
        raw_data_path: Path to raw data CSV
        processed_data_path: Path to processed wide-form CSV
        sparse_threshold: Proportion threshold for dropping sparse columns (0-1)
        feature_keywords: Keywords to match when auto-selecting features
        explicit_features: List of explicit feature column names (overrides keywords)
        target_column: Exact target column name (auto-detected if None)
    """

    raw_data_path: str = Field(
        default=os.path.join("data", "raw", "environment_tun.csv"),
        description="Path to raw data CSV file",
    )
    processed_data_path: str = Field(
        default=os.path.join("data", "processed", "processed_tunisia.csv"),
        description="Path to processed wide-form CSV file",
    )
    sparse_threshold: float = Field(
        default=0.5, description="Drop columns with more than this proportion of missing values"
    )
    feature_keywords: List[str] = Field(
        default=[
            "renewable",
            "water productivity",
            "population",
            "urban",
            "rural",
            "agricultural land",
            "arable",
            "precipitation",
            "forest",
            "greenhouse",
        ],
        description="Keywords to match when auto-selecting features",
    )
    explicit_features: Optional[List[str]] = Field(
        default=None, description="If set, use only these features (overrides keyword matching)"
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Target column name (auto-detected from 'Level of water stress' if None)",
    )

    @field_validator("sparse_threshold")
    @classmethod
    def validate_sparse_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("sparse_threshold must be between 0 and 1")
        return v

    class Config:
        extra = "forbid"
        validate_assignment = True


# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================
class FeatureEngineeringConfig(BaseModel):
    """Feature engineering parameter configuration.

    Attributes:
        lag_features: Number of lag features to add (0 for disabled)
        apply_leakage_filter: Drop features highly correlated with target
        leakage_correlation_threshold: Correlation threshold for leakage detection
        apply_collinearity_filter: Drop features with high pairwise correlation
        collinearity_threshold: Pairwise correlation threshold for filtering
        max_features: Maximum number of features to keep (0 for unlimited)
    """

    lag_features: int = Field(default=0, description="Number of lag features to add (0 to disable)")
    apply_leakage_filter: bool = Field(
        default=True, description="Drop features highly correlated with target"
    )
    leakage_correlation_threshold: float = Field(
        default=0.99, description="Absolute correlation threshold for leakage detection"
    )
    apply_collinearity_filter: bool = Field(
        default=True, description="Drop features with high pairwise correlation"
    )
    collinearity_threshold: float = Field(
        default=0.95, description="Pairwise correlation threshold for collinearity filtering"
    )
    max_features: int = Field(
        default=12, description="Maximum number of features to keep (0 for unlimited)"
    )

    @field_validator("lag_features", "max_features")
    @classmethod
    def validate_non_negative(cls, v):
        if v < 0:
            raise ValueError("Must be non-negative")
        return v

    @field_validator("leakage_correlation_threshold", "collinearity_threshold")
    @classmethod
    def validate_thresholds(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Thresholds must be between 0 and 1")
        return v

    class Config:
        extra = "forbid"
        validate_assignment = True


# ============================================================================
# TRAIN/TEST SPLIT CONFIGURATION
# ============================================================================
class SplitConfig(BaseModel):
    """Temporal train/test split configuration.

    Attributes:
        train_end_year: Last year to include in training data
    """

    train_end_year: int = Field(default=2010, description="Last year to include in training split")

    class Config:
        extra = "forbid"
        validate_assignment = True


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
class ModelConfig(BaseModel):
    """Individual model configuration with hyperparameters.

    Attributes:
        enabled: Whether to train this model
        hyperparameters: Model-specific hyperparameters as key-value pairs
    """

    enabled: bool = Field(default=True, description="Whether to train this model")
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Model-specific hyperparameters"
    )

    class Config:
        extra = "forbid"
        validate_assignment = True


class ModelsConfig(BaseModel):
    """Configuration for all models to train.

    Attributes:
        enable_hyperparameter_tuning: Use GridSearchCV to find best hyperparameters
        linear_regression: LinearRegression configuration
        decision_tree: DecisionTree configuration
        random_forest: RandomForest configuration
        ridge: Ridge regression configuration
        lasso: Lasso regression configuration
        xgboost: XGBoost configuration
    """

    enable_hyperparameter_tuning: bool = Field(
        default=False, description="Apply GridSearchCV hyperparameter tuning to all models"
    )
    linear_regression: ModelConfig = Field(
        default_factory=lambda: ModelConfig(enabled=True, hyperparameters={}),
        description="LinearRegression model configuration",
    )
    decision_tree: ModelConfig = Field(
        default_factory=lambda: ModelConfig(enabled=True, hyperparameters={"random_state": 42}),
        description="DecisionTree model configuration",
    )
    random_forest: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            enabled=True, hyperparameters={"n_estimators": 100, "random_state": 42}
        ),
        description="RandomForest model configuration",
    )
    ridge: ModelConfig = Field(
        default_factory=lambda: ModelConfig(enabled=True, hyperparameters={}),
        description="Ridge regression configuration",
    )
    lasso: ModelConfig = Field(
        default_factory=lambda: ModelConfig(enabled=True, hyperparameters={"max_iter": 10000}),
        description="Lasso regression configuration",
    )
    xgboost: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            enabled=False, hyperparameters={"random_state": 42, "verbosity": 0}
        ),
        description="XGBoost configuration",
    )

    class Config:
        extra = "forbid"
        validate_assignment = True


# ============================================================================
# EXPERIMENT TRACKING CONFIGURATION
# ============================================================================
class MLflowConfig(BaseModel):
    """MLflow experiment tracking configuration.

    Attributes:
        enabled: Enable MLflow tracking
        tracking_uri: MLflow tracking server URI (local directory path or server URL)
        experiment_name: Name of the MLflow experiment
        log_models: Save trained models to MLflow registry
        log_plots: Log feature importance and performance plots
    """

    enabled: bool = Field(default=True, description="Enable MLflow experiment tracking")
    tracking_uri: str = Field(
        default="./mlruns",
        description="MLflow tracking server URI (local path or remote server URL)",
    )
    experiment_name: str = Field(
        default="water_stress_regression", description="Name of the MLflow experiment"
    )
    log_models: bool = Field(
        default=True, description="Save trained models to MLflow model registry"
    )
    log_plots: bool = Field(
        default=True, description="Log plots (feature importance, predictions) as artifacts"
    )

    class Config:
        extra = "forbid"
        validate_assignment = True


# ============================================================================
# MAIN TRAINING CONFIGURATION
# ============================================================================
class TrainConfig(BaseModel):
    """Complete training configuration with all reproducibility guarantees.

    This is the main configuration class that brings together all settings
    needed for a reproducible ML experiment. It can be:
    - Instantiated programmatically with keyword arguments
    - Loaded from YAML file using from_yaml()
    - Serialized back to YAML using to_yaml()

    Example:
        # From YAML file
        config = TrainConfig.from_yaml("config/train_config.yaml")

        # Programmatically
        config = TrainConfig(
            seeds=SeedConfig(numpy_seed=123),
            data=DataConfig(),
            models=ModelsConfig()
        )

        # Access seeds for reproducibility
        np.random.seed(config.seeds.numpy_seed)
        random.seed(config.seeds.python_seed)
    """

    seeds: SeedConfig = Field(
        default_factory=SeedConfig, description="Random seed configuration for reproducibility"
    )
    data: DataConfig = Field(
        default_factory=DataConfig, description="Data loading and preprocessing configuration"
    )
    feature_engineering: FeatureEngineeringConfig = Field(
        default_factory=FeatureEngineeringConfig, description="Feature engineering parameters"
    )
    split: SplitConfig = Field(
        default_factory=SplitConfig, description="Train/test split configuration"
    )
    models: ModelsConfig = Field(
        default_factory=ModelsConfig, description="Models to train and their hyperparameters"
    )
    mlflow: MLflowConfig = Field(
        default_factory=MLflowConfig, description="MLflow experiment tracking configuration"
    )
    output_dir: str = Field(
        default="models_tuned", description="Directory to save model artifacts and metrics"
    )

    class Config:
        extra = "forbid"
        validate_assignment = True

    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            TrainConfig instance loaded from the YAML file

        Raises:
            FileNotFoundError: If config file does not exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If configuration validation fails

        Example:
            config = TrainConfig.from_yaml("config/train_config.yaml")
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            output_path: Path where YAML file will be saved

        Raises:
            IOError: If file cannot be written
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.model_dump(by_alias=False)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def __str__(self) -> str:
        """String representation of configuration."""
        models_enabled = sum(
            1
            for m in [
                self.models.linear_regression,
                self.models.decision_tree,
                self.models.random_forest,
                self.models.ridge,
                self.models.lasso,
                self.models.xgboost,
            ]
            if m.enabled
        )
        return (
            f"TrainConfig(\n"
            f"  {self.seeds}\n"
            f"  output_dir={self.output_dir}\n"
            f"  models_enabled={models_enabled}\n"
            f")"
        )
