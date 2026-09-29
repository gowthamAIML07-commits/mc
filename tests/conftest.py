"""Pytest configuration and global fixtures."""
import pytest
from httpx import AsyncClient, ASGITransport
from apps.backend.main import app


@pytest.fixture
async def async_client():
    """Async HTTP client fixture for API testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
