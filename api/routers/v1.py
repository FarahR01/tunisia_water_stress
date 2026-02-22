"""API v1 routers for the Tunisia Water Stress ML API.

This module organizes endpoints into logical routers for better modularity
and supports API versioning via path prefix (/api/v1/).
"""
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, BackgroundTasks

from ..config import FEATURE_COLUMNS, get_settings
from ..dependencies import ModelServiceDep, RateLimitDep
from ..exceptions import YearRangeError, PredictionError
from ..model_service import ModelService
from ..schemas import (
    HealthResponse,
    ModelsInfoResponse,
    ModelInfo,
    PredictionInput,
    PredictionOutput,
    ScenarioInput,
    ScenarioOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
    ErrorResponse
)


# ============================================================================
# System Router - Health checks and system info
# ============================================================================

system_router = APIRouter(
    prefix="",
    tags=["System"],
    responses={500: {"model": ErrorResponse}}
)


@system_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check API health status and verify models are loaded."
)
def health_check(service: ModelServiceDep) -> HealthResponse:
    """
    Check API health status.
    
    Returns the API status, version, and number of loaded models.
    """
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        models_loaded=len(service.get_available_models())
    )


# ============================================================================
# Models Router - Model information and metrics
# ============================================================================

models_router = APIRouter(
    prefix="/models",
    tags=["Models"],
    responses={500: {"model": ErrorResponse}}
)


@models_router.get(
    "",
    response_model=ModelsInfoResponse,
    summary="List Available Models",
    description="Get information about all available ML models and their performance metrics."
)
def get_models_info(service: ModelServiceDep) -> ModelsInfoResponse:
    """
    Get information about available models.
    
    Returns list of available models, their performance metrics,
    and the feature columns they expect.
    """
    available = service.get_available_models()
    metrics = service.get_model_metrics()
    
    model_metrics = [
        ModelInfo(
            name=model_name,
            mae=round(m['MAE'], 4),
            rmse=round(m['RMSE'], 4),
            r2=round(m['R2'], 4)
        )
        for model_name in available
        if (m := metrics.get(model_name))
    ]
    
    return ModelsInfoResponse(
        available_models=available,
        default_model=service.get_best_model_name(),
        metrics=model_metrics,
        feature_columns=FEATURE_COLUMNS
    )


# ============================================================================
# Predictions Router - Single and batch predictions
# ============================================================================

predictions_router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"],
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Prediction error"}
    }
)


@predictions_router.post(
    "",
    response_model=PredictionOutput,
    summary="Make Prediction",
    description="Make a single water stress prediction for a specific year."
)
def predict_water_stress(
    input_data: PredictionInput,
    service: ModelServiceDep
) -> PredictionOutput:
    """
    Make a single water stress prediction.
    
    Provide the year and optionally other features. Missing features
    will be filled from historical data or extrapolated.
    """
    prediction, model_used, features = service.predict(
        year=input_data.year,
        water_productivity=input_data.water_productivity,
        freshwater_withdrawals_agriculture=input_data.freshwater_withdrawals_agriculture,
        freshwater_withdrawals_industry=input_data.freshwater_withdrawals_industry,
        renewable_freshwater_resources=input_data.renewable_freshwater_resources,
        model_name=input_data.model_name
    )
    
    return PredictionOutput(
        year=input_data.year,
        predicted_water_stress=round(prediction, 2),
        model_used=model_used,
        input_features=features
    )


@predictions_router.post(
    "/batch",
    response_model=BatchPredictionOutput,
    summary="Batch Predictions",
    description="Submit multiple prediction requests at once (max 100)."
)
def batch_predict(
    input_data: BatchPredictionInput,
    service: ModelServiceDep
) -> BatchPredictionOutput:
    """
    Make batch water stress predictions.
    
    Submit multiple prediction requests at once for efficiency.
    Failed predictions are skipped with warnings.
    """
    results = []
    failed = 0
    
    for pred_input in input_data.predictions:
        try:
            prediction, model_used, features = service.predict(
                year=pred_input.year,
                water_productivity=pred_input.water_productivity,
                freshwater_withdrawals_agriculture=pred_input.freshwater_withdrawals_agriculture,
                freshwater_withdrawals_industry=pred_input.freshwater_withdrawals_industry,
                renewable_freshwater_resources=pred_input.renewable_freshwater_resources,
                model_name=pred_input.model_name
            )
            
            results.append(PredictionOutput(
                year=pred_input.year,
                predicted_water_stress=round(prediction, 2),
                model_used=model_used,
                input_features=features
            ))
        except Exception:
            failed += 1
    
    return BatchPredictionOutput(
        results=results,
        total_predictions=len(input_data.predictions),
        successful=len(results),
        failed=failed
    )


@predictions_router.get(
    "/years/{start_year}/{end_year}",
    response_model=list[PredictionOutput],
    summary="Predict Year Range",
    description="Get predictions for a range of years (max 100 year span)."
)
def predict_year_range(
    start_year: Annotated[int, Path(ge=1960, le=2100, description="Start year")],
    end_year: Annotated[int, Path(ge=1960, le=2100, description="End year")],
    model_name: Annotated[str | None, Query(description="Model to use")] = None,
    service: ModelServiceDep = None
) -> list[PredictionOutput]:
    """
    Get predictions for a range of years.
    
    Useful for generating time series of predictions.
    """
    settings = get_settings()
    
    if start_year > end_year:
        raise YearRangeError(start_year, end_year, settings.max_year_range)
    
    if end_year - start_year > settings.max_year_range:
        raise YearRangeError(start_year, end_year, settings.max_year_range)
    
    results = []
    for year in range(start_year, end_year + 1):
        prediction, model_used, features = service.predict(
            year=year,
            model_name=model_name
        )
        results.append(PredictionOutput(
            year=year,
            predicted_water_stress=round(prediction, 2),
            model_used=model_used,
            input_features=features
        ))
    
    return results


# ============================================================================
# Scenarios Router - Future scenario predictions
# ============================================================================

scenarios_router = APIRouter(
    prefix="/scenarios",
    tags=["Scenarios"],
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Prediction error"}
    }
)


@scenarios_router.post(
    "",
    response_model=ScenarioOutput,
    summary="Scenario Prediction",
    description="""
Generate a scenario-based prediction for a future year.

**Trend Methods:**
- `linear`: Linear extrapolation from recent trends (default)
- `exponential`: Exponential growth/decay extrapolation  
- `last_value`: Use the most recent known values
- `average`: Use historical average values
"""
)
def predict_scenario(
    input_data: ScenarioInput,
    service: ModelServiceDep
) -> ScenarioOutput:
    """
    Generate a scenario-based prediction for a future year.
    
    This endpoint extrapolates feature values based on historical trends
    and generates a prediction with interpretation.
    """
    prediction, model_used, features, interpretation = service.predict_scenario(
        target_year=input_data.target_year,
        trend_method=input_data.trend_method,
        model_name=input_data.model_name
    )
    
    return ScenarioOutput(
        target_year=input_data.target_year,
        predicted_water_stress=round(prediction, 2),
        model_used=model_used,
        trend_method=input_data.trend_method,
        extrapolated_features=features,
        interpretation=interpretation
    )


# ============================================================================
# Helper to get all routers
# ============================================================================

def get_v1_routers() -> list[APIRouter]:
    """Get all v1 API routers."""
    return [
        system_router,
        models_router,
        predictions_router,
        scenarios_router
    ]
