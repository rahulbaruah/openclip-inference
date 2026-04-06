"""
OpenCLIP model loader — singleton pattern.
Loads the model, preprocessing transforms, and tokenizer once at startup
and exposes them via module-level accessor functions.
"""

import logging
from typing import Any

import torch
import open_clip

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Module-level singletons ──────────────────────────────────────────────
_model: Any | None = None
_preprocess: Any | None = None
_tokenizer: Any | None = None
_device: torch.device | None = None


def _resolve_device(requested: str) -> torch.device:
    """Resolve the compute device from a user-friendly string."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def load_model() -> None:
    """
    Load the OpenCLIP model, preprocess pipeline, and tokenizer.
    Call this once during application startup (lifespan event).
    """
    global _model, _preprocess, _tokenizer, _device

    settings = get_settings()
    _device = _resolve_device(settings.device)

    logger.info(
        "Loading OpenCLIP model=%s  pretrained=%s  device=%s",
        settings.model_name,
        settings.pretrained,
        _device,
    )

    _model, _, _preprocess = open_clip.create_model_and_transforms(
        settings.model_name,
        pretrained=settings.pretrained,
        device=_device,
    )
    _model.eval()

    _tokenizer = open_clip.get_tokenizer(settings.model_name)

    embedding_dim = _model.visual.output_dim
    logger.info(
        "Model loaded successfully — embedding_dim=%d  device=%s",
        embedding_dim,
        _device,
    )


# ── Public accessors ─────────────────────────────────────────────────────

def get_model():
    """Return the loaded OpenCLIP model."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model


def get_preprocess():
    """Return the image preprocessing transform pipeline."""
    if _preprocess is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _preprocess


def get_tokenizer():
    """Return the text tokenizer for the loaded model."""
    if _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _tokenizer


def get_device() -> torch.device:
    """Return the resolved compute device."""
    if _device is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _device


def get_embedding_dim() -> int:
    """Return the embedding dimensionality of the loaded model."""
    return get_model().visual.output_dim


def is_model_loaded() -> bool:
    """Check whether the model has been loaded."""
    return _model is not None
