---
title: OpenCLIP Inference API
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
tags:
  - openclip
  - embeddings
  - image-embeddings
  - text-embeddings
  - clip
  - fastapi
  - docker
short_description: Self-hosted API for OpenCLIP embeddings
---

# OpenCLIP Inference API

Self-hosted REST API for generating **image and text embeddings** using [OpenCLIP](https://github.com/mlfoundations/open_clip). Designed for production use with Redis caching, Celery batch processing, and Docker deployment.

---

## Features

- 🖼️ **Image embeddings** — upload files, send base64-encoded images, or pass a public image URL
- 📝 **Text embeddings** — encode text into the same vector space as images
- 🔀 **Batch processing** — embed multiple images/texts in a single request
- ⚡ **Redis caching** — avoid recomputing embeddings for identical images
- 📦 **Celery workers** — async batch jobs for catalog-scale ingestion
- 🐳 **Docker-ready** — one command to deploy API + Redis + Celery
- 🏥 **Health checks** — built-in monitoring endpoints
- 📚 **Auto-docs** — Swagger UI at `/docs`, ReDoc at `/redoc`

---

## Quick Start

### 1. Clone & Configure

```bash
git clone <your-repo-url> openclip-inference
cd openclip-inference

# Create your environment file
cp .env.example .env
```

### 2. Run with Docker Compose

```bash
docker compose up --build -d
```

This starts three services:

| Service | Description | Port |
|---------|-------------|------|
| `openclip-api` | FastAPI + Gunicorn | `8000` |
| `openclip-redis` | Redis cache | `6379` |
| `openclip-celery` | Celery worker | — |

### 3. Verify

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info
```

---

## API Reference

### `POST /embed/image`

Generate an embedding for a single image.

**Option A: File upload**

```bash
curl -X POST http://localhost:8000/embed/image \
  -F "file=@photo.jpg"
```

**Option B: Base64 JSON**

```bash
curl -X POST http://localhost:8000/embed/image \
  -H "Content-Type: application/json" \
  -d '{"base64": "<base64-encoded-image>"}'
```

**Option C: Image URL**

Pass any publicly accessible image URL and the API will fetch and embed it server-side.

```bash
curl -X POST http://localhost:8000/embed/image \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/photo.jpg"}'
```

**Response:**

```json
{
  "embedding": [0.0123, -0.0456, ...],
  "dim": 512,
  "cached": false
}
```

---

### `POST /embed/text`

Generate an embedding for text (same vector space as images).

```bash
curl -X POST http://localhost:8000/embed/text \
  -H "Content-Type: application/json" \
  -d '{"text": "a photo of a cat"}'
```

**Response:**

```json
{
  "embedding": [0.0234, -0.0567, ...],
  "dim": 512,
  "cached": false
}
```

> **Tip:** Because image and text embeddings share the same space, you can compute cosine similarity between them for cross-modal search (e.g., search text → find matching products).

---

### `POST /embed/batch`

Embed multiple images and/or texts in one request.

```bash
curl -X POST http://localhost:8000/embed/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"type": "image", "data": "<base64-image>"},
      {"type": "text",  "data": "red sneakers"},
      {"type": "text",  "data": "blue dress"}
    ]
  }'
```

**Response:**

```json
{
  "embeddings": [
    {"index": 0, "embedding": [...], "dim": 512, "cached": false, "error": null},
    {"index": 1, "embedding": [...], "dim": 512, "cached": false, "error": null},
    {"index": 2, "embedding": [...], "dim": 512, "cached": false, "error": null}
  ],
  "count": 3,
  "errors": 0
}
```

---

### `GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "redis_connected": true,
  "device": "cpu"
}
```

### `GET /model/info`

```json
{
  "model": "ViT-B-32",
  "pretrained": "openai",
  "embedding_dim": 512,
  "device": "cpu"
}
```

---

## Configuration

All settings are configured via environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `ViT-B-32` | OpenCLIP model architecture |
| `PRETRAINED` | `openai` | Pretrained weights to use |
| `DEVICE` | `auto` | Compute device: `auto`, `cpu`, or `cuda` |
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | Bind port |
| `API_WORKERS` | `1` | Gunicorn worker count |
| `LOG_LEVEL` | `info` | Logging level |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection URL |
| `CACHE_TTL` | `86400` | Embedding cache TTL in seconds (24h) |
| `CELERY_BROKER_URL` | `redis://redis:6379/1` | Celery broker URL |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/2` | Celery result backend URL |
| `MAX_BATCH_SIZE` | `64` | Maximum items per batch request |

### Model Options

| Model | Dim | Speed | Quality | Use Case |
|-------|-----|-------|---------|----------|
| `ViT-B-32` | 512 | ⚡ Fast | Good | Default — best speed/quality balance |
| `ViT-B-16` | 512 | Medium | Better | Higher quality, moderate speed |
| `ViT-L-14` | 768 | Slow | Best | Maximum quality, needs more RAM |

---

## Local Development

### Without Docker

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API (development mode)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Run Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

### Celery Worker (local)

```bash
# Start Redis first
redis-server

# In a separate terminal
celery -A celery_worker.celery_app worker --loglevel=info
```

---

## GPU Support

To run with NVIDIA GPU acceleration:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

**Prerequisites:**
- NVIDIA Container Toolkit installed
- `nvidia-docker2` configured
- A `Dockerfile.gpu` with CUDA-enabled PyTorch (replace the CPU-only PyTorch install with the CUDA variant)

Set `DEVICE=cuda` in your `.env` file.

---

## Architecture

```
┌─────────────────┐          ┌──────────────────────────────────┐
│   Your App      │  HTTP    │   OpenCLIP Inference API         │
│   (Laravel,     │────────►│   FastAPI + Gunicorn              │
│    Node, etc.)  │          │                                  │
└─────────────────┘          │   POST /embed/image              │
                             │   POST /embed/text               │
                             │   POST /embed/batch              │
                             │   GET  /health                   │
                             │   GET  /model/info               │
                             └──────────┬───────────────────────┘
                                        │
                             ┌──────────┴───────────┐
                             │                      │
                        ┌────▼─────┐         ┌──────▼──────┐
                        │  Redis   │         │   Celery    │
                        │ (cache)  │         │  (batch)    │
                        └──────────┘         └─────────────┘
                                                    │
                                          ┌─────────▼─────────┐
                                          │ Qdrant / pgvector  │
                                          │  (vector store)    │
                                          └───────────────────┘
```

---

## Client Examples

### Python

```python
import requests

# Embed an image (file upload)
with open("photo.jpg", "rb") as f:
    resp = requests.post("http://localhost:8000/embed/image", files={"file": f})
    embedding = resp.json()["embedding"]  # list of 512 floats

# Embed an image (URL)
resp = requests.post(
    "http://localhost:8000/embed/image",
    json={"url": "https://example.com/photo.jpg"}
)
embedding = resp.json()["embedding"]

# Embed text
resp = requests.post(
    "http://localhost:8000/embed/text",
    json={"text": "red running shoes"}
)
text_embedding = resp.json()["embedding"]

# Compute similarity
import numpy as np
similarity = np.dot(embedding, text_embedding)
print(f"Similarity: {similarity:.4f}")
```

### PHP (Laravel)

```php
use Illuminate\Support\Facades\Http;

// Embed an image (file upload)
$response = Http::attach(
    'file',
    file_get_contents($imagePath),
    'photo.jpg'
)->post('http://openclip-api:8000/embed/image');

$embedding = $response->json('embedding');

// Embed an image (URL)
$response = Http::post('http://openclip-api:8000/embed/image', [
    'url' => 'https://example.com/photo.jpg',
]);

$embedding = $response->json('embedding');

// Embed text
$response = Http::post('http://openclip-api:8000/embed/text', [
    'text' => 'blue summer dress',
]);

$textEmbedding = $response->json('embedding');
```

### JavaScript / Node.js

```javascript
// Embed an image by URL
const imgResponse = await fetch('http://localhost:8000/embed/image', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ url: 'https://example.com/photo.jpg' }),
});
const { embedding: imgEmbedding, dim } = await imgResponse.json();
console.log(`Got ${dim}-dimensional image embedding`);

// Embed text
const response = await fetch('http://localhost:8000/embed/text', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'a photo of a sunset' }),
});
const { embedding, dim: textDim } = await response.json();
console.log(`Got ${textDim}-dimensional text embedding`);
```

---

## Production Checklist

- [ ] Set `API_WORKERS` based on available RAM (~1.5 GB per worker for ViT-B-32)
- [ ] Place behind a reverse proxy (Nginx / Caddy) with TLS
- [ ] Restrict access to internal VPC network or add API key auth
- [ ] Set up log aggregation (stdout → CloudWatch / Loki)
- [ ] Monitor `/health` endpoint with your uptime checker
- [ ] Tune `CACHE_TTL` based on how often your images change
- [ ] Set Redis `maxmemory` appropriately for your cache size

---

## License

MIT
