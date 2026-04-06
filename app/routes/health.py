"""
Health-check endpoint.
"""

from fastapi import APIRouter

from app.model import get_device, is_model_loaded
from app.schemas import HealthResponse
from app.services.cache import is_redis_connected

router = APIRouter(tags=["Operations"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the operational status of the API, model, and Redis.",
)
async def health_check():
    model_loaded = is_model_loaded()
    redis_ok = is_redis_connected()
    device = str(get_device()) if model_loaded else "n/a"

    status = "healthy" if model_loaded else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        redis_connected=redis_ok,
        device=device,
    )
