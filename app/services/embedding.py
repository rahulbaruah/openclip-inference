"""
Core embedding service — image and text encoding via OpenCLIP.
"""

import hashlib
import io
import logging
from typing import Optional

import torch
from PIL import Image

from app.model import get_model, get_preprocess, get_tokenizer, get_device
from app.services.cache import get_cached_embedding, set_cached_embedding

logger = logging.getLogger(__name__)


def compute_image_hash(image_bytes: bytes) -> str:
    """Compute SHA-256 hash of raw image bytes for cache keying."""
    return hashlib.sha256(image_bytes).hexdigest()


def embed_image(
    image_bytes: bytes,
    use_cache: bool = True,
) -> tuple[list[float], bool]:
    """
    Generate a normalized embedding for an image.

    Args:
        image_bytes: Raw bytes of the image file.
        use_cache: Whether to check/populate the Redis cache.

    Returns:
        (embedding, cached) — the float vector and whether it was a cache hit.
    """
    image_hash: Optional[str] = None

    # ── Check cache ──────────────────────────────────────────────────
    if use_cache:
        image_hash = compute_image_hash(image_bytes)
        cached = get_cached_embedding(image_hash)
        if cached is not None:
            logger.debug("Cache hit for image hash=%s", image_hash[:12])
            return cached, True

    # ── Encode ───────────────────────────────────────────────────────
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    preprocess = get_preprocess()
    device = get_device()

    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        embedding = get_model().encode_image(tensor)
        embedding /= embedding.norm(dim=-1, keepdim=True)  # L2 normalize

    result = embedding[0].cpu().tolist()

    # ── Populate cache ───────────────────────────────────────────────
    if use_cache and image_hash:
        set_cached_embedding(image_hash, result)

    return result, False


def embed_text(text: str) -> list[float]:
    """
    Generate a normalized embedding for a text string.

    Args:
        text: The input text to embed.

    Returns:
        Normalized float vector in the same space as image embeddings.
    """
    tokenizer = get_tokenizer()
    device = get_device()

    tokens = tokenizer([text]).to(device)

    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        embedding = get_model().encode_text(tokens)
        embedding /= embedding.norm(dim=-1, keepdim=True)

    return embedding[0].cpu().tolist()


def embed_batch(
    items: list[dict],
) -> list[dict]:
    """
    Process a batch of image/text items.

    Args:
        items: List of dicts with keys 'type' ("image"|"text") and 'data'.

    Returns:
        List of result dicts with 'embedding', 'dim', 'cached', 'error'.
    """
    from app.model import get_embedding_dim

    results = []
    dim = get_embedding_dim()

    for i, item in enumerate(items):
        try:
            if item["type"] == "image":
                import base64
                image_bytes = base64.b64decode(item["data"])
                embedding, cached = embed_image(image_bytes, use_cache=True)
                results.append({
                    "index": i,
                    "embedding": embedding,
                    "dim": dim,
                    "cached": cached,
                    "error": None,
                })
            elif item["type"] == "text":
                embedding = embed_text(item["data"])
                results.append({
                    "index": i,
                    "embedding": embedding,
                    "dim": dim,
                    "cached": False,
                    "error": None,
                })
            else:
                results.append({
                    "index": i,
                    "embedding": [],
                    "dim": dim,
                    "cached": False,
                    "error": f"Unknown item type: {item['type']}",
                })
        except Exception as exc:
            logger.warning("Batch item %d failed: %s", i, exc)
            results.append({
                "index": i,
                "embedding": [],
                "dim": dim,
                "cached": False,
                "error": str(exc),
            })

    return results
