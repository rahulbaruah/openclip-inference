"""
Pydantic schemas for API request/response models.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import AnyHttpUrl, BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────

class ItemType(str, Enum):
    image = "image"
    text = "text"


# ── Requests ─────────────────────────────────────────────────────────────

class TextEmbedRequest(BaseModel):
    """Request body for text embedding."""
    text: str = Field(..., min_length=1, max_length=10_000, description="Text to embed")


class ImageBase64Request(BaseModel):
    """Request body for base64-encoded image embedding."""
    base64: str = Field(..., min_length=1, description="Base64-encoded image data")


class ImageURLRequest(BaseModel):
    """Request body for URL-referenced image embedding."""
    url: AnyHttpUrl = Field(..., description="Publicly accessible image URL")


class BatchItem(BaseModel):
    """A single item in a batch embedding request."""
    type: ItemType
    data: str = Field(..., min_length=1, description="Base64 image data or text string")


class BatchEmbedRequest(BaseModel):
    """Request body for batch embedding."""
    items: list[BatchItem] = Field(
        ..., min_length=1, max_length=64, description="List of items to embed"
    )


# ── Responses ────────────────────────────────────────────────────────────

class EmbeddingResponse(BaseModel):
    """Response for a single embedding."""
    embedding: list[float]
    dim: int
    cached: bool = False


class BatchEmbeddingResult(BaseModel):
    """A single result within a batch response."""
    index: int
    embedding: list[float]
    dim: int
    cached: bool = False
    error: Optional[str] = None


class BatchEmbedResponse(BaseModel):
    """Response for batch embedding."""
    embeddings: list[BatchEmbeddingResult]
    count: int
    errors: int = 0


class HealthResponse(BaseModel):
    """Response for health check."""
    status: str
    model_loaded: bool
    redis_connected: bool
    device: str


class ModelInfoResponse(BaseModel):
    """Response for model info."""
    model: str
    pretrained: str
    embedding_dim: int
    device: str
