"""
Embedding endpoints:  /embed/image  |  /embed/text  |  /embed/batch
"""

import base64
import logging

import httpx

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.model import get_embedding_dim
from app.schemas import (
    BatchEmbedRequest,
    BatchEmbedResponse,
    BatchEmbeddingResult,
    EmbeddingResponse,
    ImageBase64Request,
    ImageURLRequest,
    TextEmbedRequest,
)
from app.services.embedding import embed_batch, embed_image, embed_text

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/embed", tags=["Embeddings"])

# ── Allowed MIME types ───────────────────────────────────────────────────
_ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
}


@router.post(
    "/image",
    response_model=EmbeddingResponse,
    summary="Embed an image",
    description="Upload an image file or send a base64-encoded image to get its CLIP embedding.",
)
async def embed_image_endpoint(
    file: UploadFile | None = File(None),
    body: ImageBase64Request | None = None,
    url_body: ImageURLRequest | None = None,
):
    """
    Accepts any one of:
    - A multipart file upload (`file` field), or
    - A JSON body with `{"base64": "..."}`, or
    - A JSON body with `{"url": "https://..."}` (publicly accessible image URL).
    """
    image_bytes: bytes | None = None

    # ── Multipart upload ─────────────────────────────────────────────
    if file is not None:
        if file.content_type and file.content_type not in _ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {file.content_type}. "
                       f"Allowed: {', '.join(sorted(_ALLOWED_TYPES))}",
            )
        image_bytes = await file.read()

    # ── Base64 body ──────────────────────────────────────────────────
    elif body is not None:
        try:
            image_bytes = base64.b64decode(body.base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 data")

    # ── URL body ─────────────────────────────────────────────────────
    elif url_body is not None:
        url = str(url_body.url)
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                response = await client.get(url)
            response.raise_for_status()
        except httpx.TimeoutException:
            raise HTTPException(status_code=408, detail=f"Request timed out fetching URL: {url}")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to fetch image URL (HTTP {exc.response.status_code}): {url}",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not retrieve URL: {exc}")

        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if content_type and content_type not in _ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type '{content_type}' at URL. "
                       f"Allowed: {', '.join(sorted(_ALLOWED_TYPES))}",
            )
        image_bytes = response.content

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Provide a file upload, a JSON body with 'base64', or a JSON body with 'url'.",
        )

    try:
        embedding, cached = embed_image(image_bytes)
    except Exception as exc:
        logger.error("Image embedding failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    return EmbeddingResponse(
        embedding=embedding,
        dim=get_embedding_dim(),
        cached=cached,
    )


@router.post(
    "/text",
    response_model=EmbeddingResponse,
    summary="Embed text",
    description="Get the CLIP text embedding for a given string. "
                "Returns a vector in the same space as image embeddings.",
)
async def embed_text_endpoint(body: TextEmbedRequest):
    try:
        embedding = embed_text(body.text)
    except Exception as exc:
        logger.error("Text embedding failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    return EmbeddingResponse(
        embedding=embedding,
        dim=get_embedding_dim(),
        cached=False,
    )


@router.post(
    "/batch",
    response_model=BatchEmbedResponse,
    summary="Batch embed",
    description="Submit a batch of images (base64) and/or texts for embedding.",
)
async def embed_batch_endpoint(body: BatchEmbedRequest):
    items = [{"type": item.type.value, "data": item.data} for item in body.items]

    results = embed_batch(items)

    embeddings = [BatchEmbeddingResult(**r) for r in results]
    errors = sum(1 for r in results if r["error"] is not None)

    return BatchEmbedResponse(
        embeddings=embeddings,
        count=len(embeddings),
        errors=errors,
    )
