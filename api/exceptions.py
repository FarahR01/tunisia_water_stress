"""Custom exceptions for the Tunisia Water Stress ML API.

Domain-specific exceptions with proper HTTP status codes and structured error responses.
"""
from typing import Any, Dict, Optional


class APIException(Exception):
    """Base exception for all API errors."""
    
    status_code: int = 500
    error_type: str = "api_error"
    detail: str = "An unexpected error occurred"
    
    def __init__(
        self,
        detail: Optional[str] = None,
        status_code: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.detail = detail or self.__class__.detail
        self.status_code = status_code or self.__class__.status_code
        self.headers = headers
        self.context = context or {}
        super().__init__(self.detail)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        return {
            "detail": self.detail,
            "error_type": self.error_type,
            "status_code": self.status_code,
            **self.context
        }


class NotFoundError(APIException):
    """Resource not found error."""
    
    status_code = 404
    error_type = "not_found"
    detail = "The requested resource was not found"


class ModelNotFoundError(NotFoundError):
    """Specific model not found error."""
    
    error_type = "model_not_found"
    detail = "The specified model was not found"
    
    def __init__(self, model_name: str, available_models: list[str]):
        super().__init__(
            detail=f"Model '{model_name}' not found. Available models: {', '.join(available_models)}",
            context={"model_name": model_name, "available_models": available_models}
        )


class ValidationError(APIException):
    """Validation error for invalid input."""
    
    status_code = 422
    error_type = "validation_error"
    detail = "Input validation failed"


class YearRangeError(ValidationError):
    """Year range validation error."""
    
    error_type = "year_range_error"
    
    def __init__(self, start_year: int, end_year: int, max_range: int = 100):
        if start_year > end_year:
            detail = f"start_year ({start_year}) must be <= end_year ({end_year})"
        else:
            detail = f"Year range ({end_year - start_year}) exceeds maximum allowed ({max_range})"
        super().__init__(detail=detail, context={
            "start_year": start_year,
            "end_year": end_year,
            "max_range": max_range
        })


class InvalidTrendMethodError(ValidationError):
    """Invalid trend method error."""
    
    error_type = "invalid_trend_method"
    
    VALID_METHODS = ["linear", "exponential", "last_value", "average"]
    
    def __init__(self, method: str):
        super().__init__(
            detail=f"Invalid trend method '{method}'. Valid methods: {', '.join(self.VALID_METHODS)}",
            context={"method": method, "valid_methods": self.VALID_METHODS}
        )


class PredictionError(APIException):
    """Error during prediction."""
    
    status_code = 500
    error_type = "prediction_error"
    detail = "An error occurred during prediction"
    
    def __init__(self, year: int, original_error: Optional[str] = None):
        detail = f"Error predicting for year {year}"
        if original_error:
            detail += f": {original_error}"
        super().__init__(detail=detail, context={"year": year})


class ModelLoadError(APIException):
    """Error loading ML models."""
    
    status_code = 503
    error_type = "model_load_error"
    detail = "Failed to load machine learning models"
    
    def __init__(self, models_dir: str, original_error: Optional[str] = None):
        detail = f"Failed to load models from {models_dir}"
        if original_error:
            detail += f": {original_error}"
        super().__init__(detail=detail, context={"models_dir": models_dir})


class DataNotFoundError(NotFoundError):
    """Historical data not found error."""
    
    error_type = "data_not_found"
    detail = "Historical data file not found"


class ServiceUnavailableError(APIException):
    """Service temporarily unavailable."""
    
    status_code = 503
    error_type = "service_unavailable"
    detail = "The service is temporarily unavailable"


class RateLimitExceededError(APIException):
    """Rate limit exceeded error."""
    
    status_code = 429
    error_type = "rate_limit_exceeded"
    detail = "Too many requests. Please try again later."
    
    def __init__(self, retry_after: int = 60):
        super().__init__(
            headers={"Retry-After": str(retry_after)},
            context={"retry_after_seconds": retry_after}
        )
