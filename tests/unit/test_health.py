"""Health check endpoint unit tests."""
import pytest
from httpx import AsyncClient, ASGITransport
from apps.backend.main import app


@pytest.mark.asyncio
async def test_health_check():
    """Verify that /health returns HTTP 200 and healthy status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "app_name" in data


@pytest.mark.asyncio
async def test_api_status():
    """Verify that /api/v1/status returns HTTP 200 and online status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert data["api_version"] == "v1"
