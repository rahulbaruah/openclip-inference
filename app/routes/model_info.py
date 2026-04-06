"""
Model information endpoint.
"""

from fastapi import APIRouter

from app.config import get_settings
from app.model import get_device, get_embedding_dim, is_model_loaded
from app.schemas import ModelInfoResponse

router = APIRouter(tags=["Operations"])


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model information",
    description="Returns details about the currently loaded OpenCLIP model.",
)
async def model_info():
    settings = get_settings()
    loaded = is_model_loaded()

    return ModelInfoResponse(
        model=settings.model_name,
        pretrained=settings.pretrained,
        embedding_dim=get_embedding_dim() if loaded else 0,
        device=str(get_device()) if loaded else "n/a",
    )
