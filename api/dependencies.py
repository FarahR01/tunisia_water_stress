"""Dependency injection utilities for FastAPI endpoints.

This module provides clean, testable dependencies using FastAPI's Depends system.
All dependencies are designed to be easily mockable for testing.
"""
from functools import lru_cache
from typing import Annotated, Generator
import time
from collections import defaultdict

from fastapi import Depends, Request

from .config import Settings, get_settings
from .model_service import ModelService
from .exceptions import RateLimitExceededError, ServiceUnavailableError


# ============================================================================
# Settings Dependency
# ============================================================================

def get_config() -> Settings:
    """
    Get application settings.
    
    Uses lru_cache internally for singleton behavior.
    Can be overridden in tests.
    """
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_config)]


# ============================================================================
# Model Service Dependency
# ============================================================================

_model_service_instance: ModelService | None = None


def get_model_service(settings: SettingsDep) -> ModelService:
    """
    Get or create the ModelService singleton.
    
    The service is initialized with settings from dependency injection,
    making it easy to configure and test.
    
    Args:
        settings: Application settings from DI
        
    Returns:
        ModelService instance
        
    Raises:
        ServiceUnavailableError: If the model service cannot be initialized
    """
    global _model_service_instance
    
    if _model_service_instance is None:
        try:
            _model_service_instance = ModelService(
                models_dir=settings.models_dir,
                data_path=settings.processed_data_path
            )
        except Exception as e:
            raise ServiceUnavailableError(
                detail=f"Failed to initialize model service: {str(e)}"
            )
    
    return _model_service_instance


def reset_model_service() -> None:
    """Reset the model service singleton. Useful for testing."""
    global _model_service_instance
    _model_service_instance = None


ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]


# ============================================================================
# Rate Limiting Dependency
# ============================================================================

class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        self._requests: dict[str, list[float]] = defaultdict(list)
    
    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Use X-Forwarded-For if behind proxy, otherwise use client host
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _cleanup_old_requests(self, client_id: str, current_time: float) -> None:
        """Remove requests outside the current window."""
        cutoff = current_time - self.window_seconds
        self._requests[client_id] = [
            ts for ts in self._requests[client_id] if ts > cutoff
        ]
    
    def check_rate_limit(self, request: Request) -> None:
        """
        Check if request is within rate limit.
        
        Raises:
            RateLimitExceededError: If rate limit is exceeded
        """
        client_id = self._get_client_id(request)
        current_time = time.time()
        
        self._cleanup_old_requests(client_id, current_time)
        
        if len(self._requests[client_id]) >= self.requests_per_minute:
            # Calculate retry-after based on oldest request in window
            oldest = min(self._requests[client_id])
            retry_after = int(self.window_seconds - (current_time - oldest)) + 1
            raise RateLimitExceededError(retry_after=max(1, retry_after))
        
        self._requests[client_id].append(current_time)


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter(settings: SettingsDep) -> RateLimiter:
    """Get or create the rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            requests_per_minute=settings.rate_limit_requests_per_minute
        )
    return _rate_limiter


def rate_limit_check(
    request: Request,
    settings: SettingsDep
) -> None:
    """
    Dependency that enforces rate limiting.
    
    Include in endpoint dependencies to enable rate limiting:
    
        @app.get("/endpoint", dependencies=[Depends(rate_limit_check)])
        async def endpoint():
            ...
    """
    if not settings.rate_limit_enabled:
        return
    
    limiter = get_rate_limiter(settings)
    limiter.check_rate_limit(request)


RateLimitDep = Annotated[None, Depends(rate_limit_check)]


# ============================================================================
# Request Context Dependencies
# ============================================================================

async def get_request_id(request: Request) -> str:
    """
    Get or generate a unique request ID for tracing.
    
    Checks for X-Request-ID header first, generates UUID if not present.
    """
    import uuid
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    return request_id


RequestIdDep = Annotated[str, Depends(get_request_id)]


# ============================================================================
# Common Endpoint Dependencies
# ============================================================================

def common_parameters(
    model_name: str | None = None,
) -> dict:
    """
    Common query parameters for prediction endpoints.
    
    Returns:
        Dict with common parameters
    """
    return {"model_name": model_name}


CommonParamsDep = Annotated[dict, Depends(common_parameters)]
