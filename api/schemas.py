"""Pydantic schemas for API request/response models."""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., example="healthy")
    version: str = Field(..., example="1.0.0")
    models_loaded: int = Field(..., example=5)


class ModelInfo(BaseModel):
    """Information about a single model."""
    name: str = Field(..., example="Lasso")
    mae: float = Field(..., example=4.08)
    rmse: float = Field(..., example=4.60)
    r2: float = Field(..., example=0.30)


class ModelsInfoResponse(BaseModel):
    """Response containing information about all available models."""
    available_models: List[str]
    default_model: str
    metrics: List[ModelInfo]
    feature_columns: List[str]


class PredictionInput(BaseModel):
    """Input for water stress prediction."""
    year: int = Field(..., ge=1960, le=2100, example=2025)
    water_productivity: Optional[float] = Field(
        None, 
        description="Water productivity (constant 2015 US$ GDP per cubic meter)",
        example=8.5
    )
    freshwater_withdrawals_agriculture: Optional[float] = Field(
        None,
        description="Annual freshwater withdrawals for agriculture (% of total)",
        example=75.0
    )
    freshwater_withdrawals_industry: Optional[float] = Field(
        None,
        description="Annual freshwater withdrawals for industry (% of total)",
        example=5.0
    )
    renewable_freshwater_resources: Optional[float] = Field(
        None,
        description="Renewable internal freshwater resources (billion cubic meters)",
        example=4.195
    )
    model_name: Optional[str] = Field(
        None,
        description="Model to use for prediction. Uses best model if not specified.",
        example="Lasso"
    )


class PredictionOutput(BaseModel):
    """Output from water stress prediction."""
    year: int
    predicted_water_stress: float = Field(..., description="Predicted level of water stress (%)")
    model_used: str
    input_features: Dict[str, float]
    confidence_note: str = Field(
        default="This prediction is based on historical patterns and should be interpreted with caution."
    )


class ScenarioInput(BaseModel):
    """Input for scenario-based prediction."""
    target_year: int = Field(default=2030, ge=2020, le=2100, example=2030)
    trend_method: str = Field(
        default="linear",
        description="Trend extrapolation method: 'linear', 'exponential', 'last_value', or 'average'",
        example="linear"
    )
    model_name: Optional[str] = Field(
        None,
        description="Model to use for prediction",
        example="Lasso"
    )


class ScenarioOutput(BaseModel):
    """Output from scenario prediction."""
    target_year: int
    predicted_water_stress: float
    model_used: str
    trend_method: str
    extrapolated_features: Dict[str, float]
    interpretation: str


class BatchPredictionInput(BaseModel):
    """Input for batch predictions."""
    predictions: List[PredictionInput]


class BatchPredictionOutput(BaseModel):
    """Output from batch predictions."""
    results: List[PredictionOutput]
    total_predictions: int


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str
    error_type: str = "prediction_error"
