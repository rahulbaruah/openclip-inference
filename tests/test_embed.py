"""
Tests for embedding endpoints.
"""

import base64
import math


class TestEmbedImage:
    """Tests for POST /embed/image"""

    def test_embed_image_upload(self, client, sample_image_bytes):
        """Uploading a JPEG returns a valid embedding."""
        response = client.post(
            "/embed/image",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

        data = response.json()
        assert "embedding" in data
        assert "dim" in data
        assert data["dim"] == len(data["embedding"])
        assert data["dim"] == 512  # ViT-B-32 default

    def test_embed_image_base64(self, client, sample_image_bytes):
        """Sending base64-encoded image via JSON body works."""
        b64 = base64.b64encode(sample_image_bytes).decode("utf-8")
        response = client.post(
            "/embed/image",
            json={"base64": b64},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["dim"] == 512

    def test_embed_image_normalized(self, client, sample_image_bytes):
        """Embedding is L2-normalized (unit vector)."""
        response = client.post(
            "/embed/image",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        data = response.json()
        norm = math.sqrt(sum(x ** 2 for x in data["embedding"]))
        assert abs(norm - 1.0) < 1e-4, f"L2 norm should be ~1.0, got {norm}"


class TestEmbedText:
    """Tests for POST /embed/text"""

    def test_embed_text(self, client):
        """Text embedding returns a valid vector."""
        response = client.post(
            "/embed/text",
            json={"text": "a photo of a cat"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["dim"] == 512
        assert len(data["embedding"]) == 512

    def test_embed_text_normalized(self, client):
        """Text embedding is L2-normalized."""
        response = client.post(
            "/embed/text",
            json={"text": "hello world"},
        )
        data = response.json()
        norm = math.sqrt(sum(x ** 2 for x in data["embedding"]))
        assert abs(norm - 1.0) < 1e-4

    def test_embed_text_empty_rejected(self, client):
        """Empty text is rejected with 422."""
        response = client.post("/embed/text", json={"text": ""})
        assert response.status_code == 422


class TestEmbedBatch:
    """Tests for POST /embed/batch"""

    def test_batch_mixed(self, client, sample_image_bytes):
        """Batch with mixed image + text items."""
        b64 = base64.b64encode(sample_image_bytes).decode("utf-8")
        response = client.post(
            "/embed/batch",
            json={
                "items": [
                    {"type": "image", "data": b64},
                    {"type": "text", "data": "a dog"},
                ]
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 2
        assert data["errors"] == 0
        assert len(data["embeddings"]) == 2

    def test_batch_empty_rejected(self, client):
        """Empty batch is rejected."""
        response = client.post("/embed/batch", json={"items": []})
        assert response.status_code == 422
