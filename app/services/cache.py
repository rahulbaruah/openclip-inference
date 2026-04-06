"""
Redis caching layer for embeddings.
Gracefully degrades — if Redis is unavailable, caching is silently skipped.
"""

import json
import logging
from typing import Optional

import redis

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Module-level client ──────────────────────────────────────────────────
_redis_client: Optional[redis.Redis] = None
_redis_available: bool = False


def init_redis() -> None:
    """
    Initialize the Redis connection pool.
    Called during application startup.
    """
    global _redis_client, _redis_available

    settings = get_settings()
    try:
        _redis_client = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
            retry_on_timeout=True,
        )
        _redis_client.ping()
        _redis_available = True
        logger.info("Redis connected at %s", settings.redis_url)
    except Exception as exc:
        _redis_available = False
        logger.warning("Redis unavailable — caching disabled: %s", exc)


def is_redis_connected() -> bool:
    """Check if Redis is reachable (cached state + live ping)."""
    if not _redis_available or _redis_client is None:
        return False
    try:
        _redis_client.ping()
        return True
    except Exception:
        return False


def _cache_key(image_hash: str) -> str:
    """Build a namespaced cache key."""
    return f"openclip:emb:{image_hash}"


def get_cached_embedding(image_hash: str) -> Optional[list[float]]:
    """
    Retrieve a cached embedding by image hash.
    Returns None on cache miss or if Redis is unavailable.
    """
    if not _redis_available or _redis_client is None:
        return None

    try:
        raw = _redis_client.get(_cache_key(image_hash))
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.debug("Cache read failed: %s", exc)
        return None


def set_cached_embedding(
    image_hash: str,
    embedding: list[float],
    ttl: Optional[int] = None,
) -> None:
    """
    Store an embedding in Redis with an optional TTL.
    Silently skips if Redis is unavailable.
    """
    if not _redis_available or _redis_client is None:
        return

    if ttl is None:
        ttl = get_settings().cache_ttl

    try:
        _redis_client.setex(
            _cache_key(image_hash),
            ttl,
            json.dumps(embedding),
        )
    except Exception as exc:
        logger.debug("Cache write failed: %s", exc)
