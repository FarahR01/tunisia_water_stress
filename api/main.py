"""FastAPI application for Tunisia Water Stress ML predictions."""
from typing import List
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config import API_TITLE, API_VERSION, API_DESCRIPTION, FEATURE_COLUMNS
from .schemas import (
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
from .model_service import get_model_service, ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model service on startup."""
    print("🚀 Starting Tunisia Water Stress ML API...")
    # Initialize model service (loads models)
    try:
        service = get_model_service()
        print(f"✓ Loaded {len(service.get_available_models())} models")
    except Exception as e:
        print(f"✗ Failed to load models: {e}")
        raise
    yield
    print("👋 Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service() -> ModelService:
    """Dependency to get the model service."""
    return get_model_service()


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {"message": "Welcome to Tunisia Water Stress ML API. Visit /docs for documentation."}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(service: ModelService = Depends(get_service)):
    """
    Check API health status.
    
    Returns the API status, version, and number of loaded models.
    """
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        models_loaded=len(service.get_available_models())
    )


@app.get("/models", response_model=ModelsInfoResponse, tags=["Models"])
async def get_models_info(service: ModelService = Depends(get_service)):
    """
    Get information about available models.
    
    Returns list of available models, their performance metrics,
    and the feature columns they expect.
    """
    available = service.get_available_models()
    metrics = service.get_model_metrics()
    
    model_metrics = []
    for model_name in available:
        if model_name in metrics:
            m = metrics[model_name]
            model_metrics.append(ModelInfo(
                name=model_name,
                mae=round(m['MAE'], 4),
                rmse=round(m['RMSE'], 4),
                r2=round(m['R2'], 4)
            ))
    
    return ModelsInfoResponse(
        available_models=available,
        default_model=service.get_best_model_name(),
        metrics=model_metrics,
        feature_columns=FEATURE_COLUMNS
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict_water_stress(
    input_data: PredictionInput,
    service: ModelService = Depends(get_service)
):
    """
    Make a single water stress prediction.
    
    Provide the year and optionally other features. Missing features
    will be filled from historical data or extrapolated.
    """
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Predictions"])
async def batch_predict(
    input_data: BatchPredictionInput,
    service: ModelService = Depends(get_service)
):
    """
    Make batch water stress predictions.
    
    Submit multiple prediction requests at once for efficiency.
    """
    results = []
    
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
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error predicting for year {pred_input.year}: {str(e)}"
            )
    
    return BatchPredictionOutput(
        results=results,
        total_predictions=len(results)
    )


@app.post("/predict/scenario", response_model=ScenarioOutput, tags=["Scenarios"])
async def predict_scenario(
    input_data: ScenarioInput,
    service: ModelService = Depends(get_service)
):
    """
    Generate a scenario-based prediction for a future year.
    
    This endpoint extrapolates feature values based on historical trends
    and generates a prediction with interpretation.
    
    **Trend Methods:**
    - `linear`: Linear extrapolation from recent trends (default)
    - `exponential`: Exponential growth/decay extrapolation
    - `last_value`: Use the most recent known values
    - `average`: Use historical average values
    """
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/years/{start_year}/{end_year}", response_model=List[PredictionOutput], tags=["Predictions"])
async def predict_year_range(
    start_year: int,
    end_year: int,
    model_name: str = None,
    service: ModelService = Depends(get_service)
):
    """
    Get predictions for a range of years.
    
    Useful for generating time series of predictions.
    """
    if start_year > end_year:
        raise HTTPException(status_code=400, detail="start_year must be <= end_year")
    
    if end_year - start_year > 100:
        raise HTTPException(status_code=400, detail="Year range cannot exceed 100 years")
    
    results = []
    for year in range(start_year, end_year + 1):
        try:
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
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error predicting for year {year}: {str(e)}")
    
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
