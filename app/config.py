"""
Application configuration via environment variables.
Uses pydantic-settings for validation and type coercion.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configurable settings for the OpenCLIP Inference API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    # ── API ──────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    log_level: str = "info"

    # ── Redis ────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 86400  # seconds (24 h)

    # ── Celery ───────────────────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── Batch ────────────────────────────────────────────────────────────
    max_batch_size: int = 64


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — parsed once, reused everywhere."""
    return Settings()
