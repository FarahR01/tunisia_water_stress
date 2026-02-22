"""Tunisia Water Stress ML API Package.

This package provides a RESTful API for water stress predictions using
trained machine learning models.

Modules:
- main: FastAPI application and entry point
- config: Configuration settings with environment support
- schemas: Pydantic models for request/response validation
- model_service: ML model loading and prediction service
- dependencies: FastAPI dependency injection utilities
- exceptions: Custom exception classes
- logging_config: Structured logging configuration
- routers: Versioned API endpoint routers
"""

from .config import get_settings, Settings
from .main import app, create_application

__all__ = [
    "app",
    "create_application", 
    "get_settings",
    "Settings",
]

__version__ = "1.0.0"
