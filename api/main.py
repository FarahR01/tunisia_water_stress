"""FastAPI application for Tunisia Water Stress ML predictions.

This is the main application entry point that:
- Configures the FastAPI application with best practices
- Sets up middleware (CORS, logging, rate limiting)
- Registers exception handlers
- Includes versioned API routers
- Manages application lifecycle
"""
import logging
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from .config import get_settings, Settings
from .dependencies import get_model_service, reset_model_service, SettingsDep
from .exceptions import (
    APIException,
    ModelNotFoundError,
    ValidationError,
    PredictionError,
    ServiceUnavailableError,
    RateLimitExceededError,
)
from .logging_config import setup_logging, LoggingMiddleware, get_logger
from .routers import get_v1_routers
from .schemas import ErrorResponse


logger = get_logger("api.main")


# ============================================================================
# Application Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Startup: Initialize logging, load models
    - Shutdown: Cleanup resources
    """
    settings = get_settings()
    
    # Setup logging
    setup_logging(
        log_level=settings.log_level,
        log_format=settings.log_format
    )
    
    logger.info(f"Starting {settings.api_title} v{settings.api_version}")
    
    # Initialize model service (pre-load models)
    try:
        service = get_model_service(settings)
        logger.info(f"Loaded {len(service.get_available_models())} models")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")
    reset_model_service()


# ============================================================================
# Application Factory
# ============================================================================

def create_application() -> FastAPI:
    """
    Application factory function.
    
    Creates and configures the FastAPI application with all middleware,
    exception handlers, and routers. This pattern supports testing
    and multiple configurations.
    
    Returns:
        Configured FastAPI application
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        responses={
            400: {"model": ErrorResponse, "description": "Bad request"},
            404: {"model": ErrorResponse, "description": "Not found"},
            422: {"model": ErrorResponse, "description": "Validation error"},
            429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
            500: {"model": ErrorResponse, "description": "Internal server error"},
            503: {"model": ErrorResponse, "description": "Service unavailable"},
        }
    )
    
    # Register middleware
    _setup_middleware(app, settings)
    
    # Register exception handlers
    _setup_exception_handlers(app, settings)
    
    # Register routers
    _setup_routers(app, settings)
    
    return app


def _setup_middleware(app: FastAPI, settings: Settings) -> None:
    """Configure application middleware."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    
    # Logging middleware
    app.add_middleware(LoggingMiddleware)


def _setup_exception_handlers(app: FastAPI, settings: Settings) -> None:
    """Register global exception handlers."""
    
    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
        """Handle all custom API exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
            headers=exc.headers,
        )
    
    @app.exception_handler(PydanticValidationError)
    async def pydantic_validation_handler(
        request: Request, 
        exc: PydanticValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors."""
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Request validation failed",
                "error_type": "validation_error",
                "errors": exc.errors(),
            },
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, 
        exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.exception(f"Unhandled exception: {exc}")
        
        # In debug mode, include exception details
        if settings.api_debug:
            detail = f"{type(exc).__name__}: {str(exc)}"
        else:
            detail = "An unexpected error occurred"
        
        return JSONResponse(
            status_code=500,
            content={
                "detail": detail,
                "error_type": "internal_error",
            },
        )


def _setup_routers(app: FastAPI, settings: Settings) -> None:
    """Register API routers."""
    
    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint redirect to docs."""
        return {
            "message": f"Welcome to {settings.api_title}",
            "version": settings.api_version,
            "docs": "/docs",
            "health": f"{settings.api_prefix}/health"
        }
    
    # Include v1 routers with prefix
    for router in get_v1_routers():
        app.include_router(router, prefix=settings.api_prefix)


# ============================================================================
# Application Instance
# ============================================================================

# Create the application instance
app = create_application()


# ============================================================================
# Development Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.api_debug,
        log_level=settings.log_level.lower(),
    )
