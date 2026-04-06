"""
FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.model import load_model
from app.routes import embed, health, model_info
from app.services.cache import init_redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    settings = get_settings()
    logger.info(
        "Starting OpenCLIP Inference API — model=%s  pretrained=%s",
        settings.model_name,
        settings.pretrained,
    )

    # Load model (blocks until weights are in memory)
    load_model()

    # Connect Redis (non-blocking, graceful if unavailable)
    init_redis()

    yield  # ← app is running

    logger.info("Shutting down OpenCLIP Inference API")


# ── App ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OpenCLIP Inference API",
    description=(
        "Self-hosted REST API for generating image and text embeddings "
        "using OpenCLIP. Supports single and batch requests with Redis caching."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Routes ───────────────────────────────────────────────────────────────
app.include_router(embed.router)
app.include_router(health.router)
app.include_router(model_info.router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "OpenCLIP Inference API",
        "version": "1.0.0",
        "docs": "/docs",
    }
