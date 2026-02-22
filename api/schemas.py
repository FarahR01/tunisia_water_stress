"""Pydantic schemas for API request/response models.

This module defines all request and response models with:
- Strong type hints
- Field validation with meaningful constraints
- Rich examples for OpenAPI documentation
- Model configuration for performance and safety
"""

from typing import Annotated, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# ============================================================================
# Base Configuration
# ============================================================================


class StrictBaseModel(BaseModel):
    """Base model with strict configuration for all schemas."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",  # Reject unknown fields
        json_schema_extra={"additionalProperties": False},
    )


class ResponseBaseModel(BaseModel):
    """Base model for response schemas (more lenient)."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )


# ============================================================================
# System Schemas
# ============================================================================


class HealthResponse(ResponseBaseModel):
    """Health check response indicating API status."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Current health status of the API",
        json_schema_extra={"example": "healthy"},
    )
    version: str = Field(..., description="API version", json_schema_extra={"example": "1.0.0"})
    models_loaded: int = Field(
        ...,
        ge=0,
        description="Number of ML models currently loaded",
        json_schema_extra={"example": 5},
    )


class ErrorResponse(ResponseBaseModel):
    """Structured error response for API errors."""

    detail: str = Field(
        ...,
        description="Human-readable error description",
        json_schema_extra={"example": "Model 'InvalidModel' not found"},
    )
    error_type: str = Field(
        default="api_error",
        description="Machine-readable error type for client handling",
        json_schema_extra={"example": "model_not_found"},
    )
    status_code: int = Field(
        default=500,
        ge=400,
        le=599,
        description="HTTP status code",
        json_schema_extra={"example": 404},
    )


# ============================================================================
# Model Information Schemas
# ============================================================================


class ModelInfo(ResponseBaseModel):
    """Performance metrics for a single model."""

    name: str = Field(..., description="Model name", json_schema_extra={"example": "Lasso"})
    mae: float = Field(
        ...,
        ge=0,
        description="Mean Absolute Error on test set",
        json_schema_extra={"example": 4.08},
    )
    rmse: float = Field(
        ...,
        ge=0,
        description="Root Mean Square Error on test set",
        json_schema_extra={"example": 4.60},
    )
    r2: float = Field(
        ...,
        le=1.0,
        description="R-squared coefficient of determination",
        json_schema_extra={"example": 0.30},
    )


class ModelsInfoResponse(ResponseBaseModel):
    """Response containing all available models and their metrics."""

    available_models: list[str] = Field(..., description="List of available model names")
    default_model: str = Field(..., description="Default model used when none specified")
    metrics: list[ModelInfo] = Field(..., description="Performance metrics for each model")
    feature_columns: list[str] = Field(..., description="Feature columns expected by the models")


# ============================================================================
# Prediction Input Schemas
# ============================================================================


class PredictionInput(StrictBaseModel):
    """Input schema for water stress prediction requests."""

    year: int = Field(
        ...,
        ge=1960,
        le=2100,
        description="Year for prediction (1960-2100)",
        json_schema_extra={"example": 2025},
    )
    water_productivity: float | None = Field(
        None,
        ge=0,
        le=100,
        description="Water productivity (constant 2015 US$ GDP per cubic meter). Range: 0-100.",
        json_schema_extra={"example": 8.5},
    )
    freshwater_withdrawals_agriculture: float | None = Field(
        None,
        ge=0,
        le=100,
        description="Annual freshwater withdrawals for agriculture (% of total). Range: 0-100.",
        json_schema_extra={"example": 75.0},
    )
    freshwater_withdrawals_industry: float | None = Field(
        None,
        ge=0,
        le=100,
        description="Annual freshwater withdrawals for industry (% of total). Range: 0-100.",
        json_schema_extra={"example": 5.0},
    )
    renewable_freshwater_resources: float | None = Field(
        None,
        ge=0,
        le=1000,
        description="Renewable internal freshwater resources (billion cubic meters). Range: 0-1000.",
        json_schema_extra={"example": 4.195},
    )
    model_name: str | None = Field(
        None,
        max_length=50,
        description="Model to use for prediction. Uses best performing model if not specified.",
        json_schema_extra={"example": "Lasso"},
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str | None) -> str | None:
        """Strip and validate model name."""
        if v is not None:
            v = v.strip()
            if len(v) == 0:
                return None
        return v

    @model_validator(mode="after")
    def validate_withdrawal_percentages(self):
        """Validate that withdrawal percentages don't exceed 100% together."""
        agri = self.freshwater_withdrawals_agriculture
        indus = self.freshwater_withdrawals_industry

        if agri is not None and indus is not None:
            total = agri + indus
            if total > 100:
                raise ValueError(
                    f"Combined agriculture ({agri}%) and industry ({indus}%) "
                    f"withdrawals ({total}%) cannot exceed 100%"
                )
        return self


class BatchPredictionInput(StrictBaseModel):
    """Input schema for batch prediction requests."""

    predictions: list[PredictionInput] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of prediction requests (max 100 per batch)",
    )


class ScenarioInput(StrictBaseModel):
    """Input schema for scenario-based predictions."""

    target_year: int = Field(
        default=2030,
        ge=2020,
        le=2100,
        description="Target year for scenario prediction (2020-2100)",
        json_schema_extra={"example": 2030},
    )
    trend_method: Literal["linear", "exponential", "last_value", "average"] = Field(
        default="linear",
        description="Method for trend extrapolation",
        json_schema_extra={"example": "linear"},
    )
    model_name: str | None = Field(
        None,
        max_length=50,
        description="Model to use for prediction",
        json_schema_extra={"example": "Lasso"},
    )


# ============================================================================
# Prediction Output Schemas
# ============================================================================


class PredictionOutput(ResponseBaseModel):
    """Output schema for a single prediction result."""

    year: int = Field(..., description="Year of prediction")
    predicted_water_stress: float = Field(..., description="Predicted water stress level (%)")
    model_used: str = Field(..., description="Name of model used for prediction")
    input_features: dict[str, float] = Field(
        ..., description="Input feature values used for prediction"
    )
    confidence_note: str = Field(
        default="This prediction is based on historical patterns and should be interpreted with caution.",
        description="Note about prediction confidence",
    )


class BatchPredictionOutput(ResponseBaseModel):
    """Output schema for batch prediction results."""

    results: list[PredictionOutput] = Field(..., description="List of prediction results")
    total_predictions: int = Field(..., ge=0, description="Total number of predictions made")
    successful: int = Field(default=0, ge=0, description="Number of successful predictions")
    failed: int = Field(default=0, ge=0, description="Number of failed predictions")


class ScenarioOutput(ResponseBaseModel):
    """Output schema for scenario prediction results."""

    target_year: int = Field(..., description="Target year of scenario")
    predicted_water_stress: float = Field(..., description="Predicted water stress level (%)")
    model_used: str = Field(..., description="Name of model used")
    trend_method: str = Field(..., description="Trend extrapolation method used")
    extrapolated_features: dict[str, float] = Field(
        ..., description="Extrapolated feature values used"
    )
    interpretation: str = Field(..., description="Human-readable interpretation of the prediction")


# ============================================================================
# Year Range Query Schemas
# ============================================================================


class YearRangeParams(StrictBaseModel):
    """Parameters for year range queries."""

    start_year: int = Field(..., ge=1960, le=2100, description="Start year of range")
    end_year: int = Field(..., ge=1960, le=2100, description="End year of range")
    model_name: str | None = Field(None, description="Model to use for predictions")

    @model_validator(mode="after")
    def validate_range(self):
        """Validate year range."""
        if self.start_year > self.end_year:
            raise ValueError(
                f"start_year ({self.start_year}) must be <= end_year ({self.end_year})"
            )
        if self.end_year - self.start_year > 100:
            raise ValueError("Year range cannot exceed 100 years")
        return self
