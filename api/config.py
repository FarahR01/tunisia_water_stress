"""API Configuration settings using Pydantic Settings for validation and environment support.

This module provides type-safe configuration with automatic environment variable loading,
validation, and sensible defaults. Use `get_settings()` for cached singleton access.
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    Settings can be overridden via environment variables:
    - API_TITLE, API_VERSION, API_DEBUG, etc.
    - MODELS_DIR, PROCESSED_DATA_PATH
    - RATE_LIMIT_ENABLED, RATE_LIMIT_REQUESTS_PER_MINUTE
    
    Example:
        export API_DEBUG=true
        export RATE_LIMIT_REQUESTS_PER_MINUTE=100
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Settings
    api_title: str = Field(
        default="Tunisia Water Stress ML API",
        description="API title for OpenAPI docs"
    )
    api_version: str = Field(
        default="1.0.0",
        description="API version string"
    )
    api_debug: bool = Field(
        default=False,
        description="Enable debug mode (shows stack traces)"
    )
    api_prefix: str = Field(
        default="/api/v1",
        description="API route prefix for versioning"
    )
    
    # CORS Settings
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins. Use specific origins in production."
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    rate_limit_requests_per_minute: int = Field(
        default=60,
        ge=1,
        le=10000,
        description="Maximum requests per minute per client"
    )
    
    # Model Settings
    models_dir: Path = Field(
        default=PROJECT_ROOT / "models_tuned",
        description="Directory containing trained models"
    )
    default_model_name: str = Field(
        default="Lasso",
        description="Default model to use for predictions"
    )
    available_models: list[str] = Field(
        default=["LinearRegression", "DecisionTree", "RandomForest", "Ridge", "Lasso"],
        description="List of available model names"
    )
    
    # Data Settings
    processed_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "processed" / "processed_tunisia.csv",
        description="Path to processed data file"
    )
    
    # Logging Settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: Literal["json", "console"] = Field(
        default="console",
        description="Log output format"
    )
    
    # Validation year range
    min_year: int = Field(default=1960, description="Minimum valid year for predictions")
    max_year: int = Field(default=2100, description="Maximum valid year for predictions")
    max_year_range: int = Field(default=100, description="Maximum year range for batch queries")
    
    @field_validator("models_dir", "processed_data_path", mode="before")
    @classmethod
    def resolve_path(cls, v):
        """Resolve paths relative to project root if needed."""
        if isinstance(v, str):
            path = Path(v)
            if not path.is_absolute():
                return PROJECT_ROOT / path
            return path
        return v
    
    @computed_field
    @property
    def api_description(self) -> str:
        """Generate API description for OpenAPI docs."""
        return """
## Tunisia Water Stress Prediction API

This API provides predictions for water stress levels in Tunisia using machine learning models.

### Features:
- **Health Check**: Verify API status and model availability
- **Model Info**: Get information about available models and their performance metrics
- **Predictions**: Make water stress predictions for specific years
- **Scenario Analysis**: Generate future scenario predictions with trend extrapolation
- **Batch Predictions**: Submit multiple predictions in a single request

### Rate Limiting:
Requests are limited to {} requests per minute per client.

### Authentication:
Currently this API is open. Future versions may require API keys.
""".format(self.rate_limit_requests_per_minute)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure singleton behavior and avoid
    repeated environment variable parsing.
    
    Returns:
        Settings instance
    """
    return Settings()


# Feature columns expected by the trained models
# These are the 5 features used after collinearity filtering
FEATURE_COLUMNS: list[str] = [
    "year",
    "Water productivity, total (constant 2015 US$ GDP per cubic meter of total freshwater withdrawal)",
    "Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)",
    "Annual freshwater withdrawals, industry (% of total freshwater withdrawal)",
    "Renewable internal freshwater resources, total (billion cubic meters)"
]

# Target variable
TARGET_COLUMN: str = "Level of water stress: freshwater withdrawal as a proportion of available freshwater resources"

# Valid trend methods for extrapolation
VALID_TREND_METHODS: list[str] = ["linear", "exponential", "last_value", "average"]


# Backwards compatibility - expose commonly used settings
# These will use defaults if Settings not yet configured
_settings = get_settings()
API_TITLE = _settings.api_title
API_VERSION = _settings.api_version
API_DESCRIPTION = _settings.api_description
