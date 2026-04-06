"""
Celery batch embedding tasks.
"""

import logging
from typing import Any

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    name="openclip.embed_batch",
    max_retries=2,
    default_retry_delay=10,
    soft_time_limit=300,
    time_limit=600,
)
def process_batch_task(self, items: list[dict[str, Any]]) -> dict:
    """
    Celery task for asynchronous batch embedding.

    Args:
        items: List of dicts with 'type' ("image"|"text") and 'data' keys.
               Images should be base64-encoded.

    Returns:
        Dict with 'results' (list of embedding dicts) and metadata.
    """
    from app.model import load_model, is_model_loaded
    from app.services.embedding import embed_batch

    # Ensure model is loaded in the worker process
    if not is_model_loaded():
        logger.info("Loading model in Celery worker...")
        load_model()

    logger.info("Processing batch of %d items (task_id=%s)", len(items), self.request.id)

    try:
        results = embed_batch(items)

        errors = sum(1 for r in results if r.get("error") is not None)

        return {
            "task_id": self.request.id,
            "results": results,
            "count": len(results),
            "errors": errors,
            "status": "completed",
        }

    except Exception as exc:
        logger.error("Batch task %s failed: %s", self.request.id, exc)
        raise self.retry(exc=exc)
