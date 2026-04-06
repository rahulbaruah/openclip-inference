# ═══════════════════════════════════════════════════════════════════════
# OpenCLIP Inference API — Multi-stage Dockerfile
# ═══════════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ─────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch to keep image small (~2 GB vs ~6 GB with CUDA)
RUN pip install --no-cache-dir --prefix=/install \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# System deps for Pillow and Redis
RUN apt-get update && \
    apt-get install -y --no-install-recommends libjpeg62-turbo libwebp7 redis-server && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Non-root user
RUN groupadd -r appuser && useradd -m -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Pre-download model weights at build time for fast cold starts
RUN python -c "\
import open_clip; \
open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')" \
    && echo "Model weights cached successfully"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

# Production server via Gunicorn + Uvicorn workers
CMD ["sh", "-c", "redis-server --daemonize yes && gunicorn app.main:app -c gunicorn.conf.py"]
