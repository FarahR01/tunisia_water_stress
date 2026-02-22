"""API routers package.

Organizes endpoints into versioned routers for better modularity
and API versioning support.
"""
from .v1 import (
    system_router,
    models_router,
    predictions_router,
    scenarios_router,
    get_v1_routers
)

__all__ = [
    "system_router",
    "models_router", 
    "predictions_router",
    "scenarios_router",
    "get_v1_routers"
]
