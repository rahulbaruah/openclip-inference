"""
Tests for operational endpoints.
"""


class TestHealth:
    """Tests for GET /health"""

    def test_health_returns_status(self, client):
        """Health endpoint returns expected fields."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "device" in data
        assert "redis_connected" in data


class TestModelInfo:
    """Tests for GET /model/info"""

    def test_model_info(self, client):
        """Model info returns correct model details."""
        response = client.get("/model/info")
        assert response.status_code == 200

        data = response.json()
        assert data["model"] == "ViT-B-32"
        assert data["pretrained"] == "openai"
        assert data["embedding_dim"] == 512
        assert "device" in data


class TestRoot:
    """Tests for GET /"""

    def test_root(self, client):
        """Root returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["service"] == "OpenCLIP Inference API"
