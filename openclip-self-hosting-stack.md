# Self-Hosting OpenCLIP: Image Embedding API — Tech Stack Guide

## Overview

This document outlines the recommended architecture and tech stack for self-hosting OpenCLIP and serving image vector embeddings via a REST API.

---

## Core ML Service (Python)

| Component | Choice | Reason |
|-----------|--------|--------|
| API Framework | **FastAPI** | Async, high-performance, already in your stack |
| Embedding Model | **OpenCLIP** (`open_clip_torch`) | Open-source CLIP implementation |
| ML Backend | **PyTorch** | Required backend for OpenCLIP |
| Image Processing | **Pillow / torchvision** | Image preprocessing pipeline |
| ASGI Server | **Uvicorn / Gunicorn** | Production-grade serving |

### Recommended Model

- **ViT-B/32** with `openai` pretrained weights — fast, well-tested, 512-dim output
- **ViT-L/14** — slower, 768-dim, higher quality (upgrade when needed)

---

## API Design

```
POST /embed/image        → accepts image file or base64, returns float[]
POST /embed/text         → accepts text string, returns float[] (same space!)
POST /embed/batch        → batch of images/texts
GET  /health
GET  /model/info
```

> **Note:** Returning both image and text embeddings from the same model enables cross-modal similarity search — very useful for e-commerce (e.g., search by text → find visually similar products).

---

## Caching & Queue

| Component | Role |
|-----------|------|
| **Redis** | Cache embeddings by image hash (avoid recomputing for the same image) |
| **Celery** | Async/batch embedding jobs (already in your stack) |

- Use **sync endpoint** for single real-time image embeds
- Use **Celery tasks** for bulk product catalog ingestion triggered from Laravel

---

## Quick Start Code

```python
from fastapi import FastAPI, UploadFile
import open_clip
import torch
from PIL import Image
import io

app = FastAPI()

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model.eval()

@app.post("/embed/image")
async def embed_image(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(tensor)
        embedding /= embedding.norm(dim=-1, keepdim=True)  # normalize
    return {"embedding": embedding[0].tolist(), "dim": embedding.shape[-1]}
```

---

## Architecture Diagram

```
Laravel (EC2) ──► FastAPI OpenCLIP Service (EC2, internal VPC)
                              │
                      ┌───────┴────────┐
                    Redis           Celery Worker
                   (cache)          (batch jobs)
                              │
                     Qdrant / pgvector
                       (vector store)
```

---

## Key Decisions

| Decision | Options | Recommendation |
|----------|---------|----------------|
| **Model size** | ViT-B/32 vs ViT-L/14 | Start with ViT-B/32 (fast, 512-dim); upgrade to ViT-L/14 for quality |
| **Request mode** | Sync vs Async | Sync for single embeds, Celery for bulk catalog ingestion |

---

## Recommended Starting Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI + Uvicorn |
| Model | OpenCLIP ViT-B/32 |
| Cache | Redis (embedding cache by image hash) |
| Queue | Celery (existing) |

