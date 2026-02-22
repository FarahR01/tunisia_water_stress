"""Test configuration and fixtures for API tests.

This module provides pytest fixtures for:
- Test client setup
- Mock model service
- Test settings
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from typing import Generator

from fastapi.testclient import TestClient

from api.config import Settings, get_settings
from api.model_service import ModelService
from api.dependencies import reset_model_service


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings."""
    return Settings(
        api_title="Test API",
        api_version="1.0.0-test",
        api_debug=True,
        api_prefix="/api/v1",
        rate_limit_enabled=False,
        log_level="DEBUG",
        log_format="console",
    )


@pytest.fixture
def mock_model_service() -> Mock:
    """Create a mock model service."""
    service = Mock(spec=ModelService)

    # Configure default return values
    service.get_available_models.return_value = ["Lasso", "Ridge", "RandomForest"]
    service.get_best_model_name.return_value = "Lasso"
    service.get_model_metrics.return_value = {
        "Lasso": {"MAE": 4.08, "RMSE": 4.60, "R2": 0.30},
        "Ridge": {"MAE": 4.20, "RMSE": 4.80, "R2": 0.25},
        "RandomForest": {"MAE": 5.50, "RMSE": 6.10, "R2": 0.15},
    }

    # Configure predict to return realistic values
    service.predict.return_value = (
        85.5,  # prediction
        "Lasso",  # model_used
        {  # features
            "year": 2025,
            "water_productivity": 8.5,
            "freshwater_withdrawals_agriculture": 75.0,
            "freshwater_withdrawals_industry": 5.0,
            "renewable_freshwater_resources": 4.195,
        },
    )

    # Configure predict_scenario
    service.predict_scenario.return_value = (
        90.0,  # prediction
        "Lasso",  # model_used
        {  # features
            "year": 2030,
            "water_productivity": 9.0,
            "freshwater_withdrawals_agriculture": 70.0,
            "freshwater_withdrawals_industry": 6.0,
            "renewable_freshwater_resources": 4.0,
        },
        "By 2030, Tunisia is predicted to have high water stress at 90.0%.",
    )

    return service


@pytest.fixture
def test_client(
    test_settings: Settings, mock_model_service: Mock
) -> Generator[TestClient, None, None]:
    """Create a test client with mocked dependencies."""
    # Reset any existing service
    reset_model_service()

    # Patch at module level where they're imported
    with patch("api.dependencies._model_service_instance", mock_model_service):
        with patch("api.dependencies.get_model_service", return_value=mock_model_service):
            with patch("api.config.get_settings", return_value=test_settings):
                with patch("api.main.get_settings", return_value=test_settings):
                    with patch("api.routers.v1.get_settings", return_value=test_settings):
                        # Import and create app after patching
                        from api.main import create_application

                        app = create_application()

                        with TestClient(app) as client:
                            yield client

    # Cleanup
    reset_model_service()


@pytest.fixture
def api_prefix(test_settings: Settings) -> str:
    """Get the API prefix for test requests."""
    return test_settings.api_prefix
