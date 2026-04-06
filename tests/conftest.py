"""
Pytest fixtures for OpenCLIP API tests.
"""

import os
import pytest
from fastapi.testclient import TestClient

# Override settings BEFORE importing the app
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/14")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/13")
os.environ.setdefault("DEVICE", "cpu")


@pytest.fixture(scope="session")
def client():
    """
    Session-scoped test client.
    The model loads once for the entire test suite.
    """
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Generate a minimal valid JPEG image for testing."""
    from PIL import Image
    import io

    img = Image.new("RGB", (64, 64), color=(128, 64, 192))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
