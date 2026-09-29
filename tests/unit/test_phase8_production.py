"""Unit Tests for Phase 8 Production Hardening, Security Headers, and Configuration."""
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from apps.backend.main import app
from database.connection import check_database_health

client = TestClient(app)


def test_security_headers_applied_on_all_responses():
    """Verify production security headers are attached to responses."""
    response = client.get("/health")
    assert response.status_code == 200
    headers = response.headers

    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("X-XSS-Protection") == "1; mode=block"
    assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "Content-Security-Policy" in headers
    assert "default-src 'self'" in headers["Content-Security-Policy"]


def test_correlation_id_tracking():
    """Verify unique correlation ID is propagated across requests."""
    # Test auto-generated correlation ID
    res1 = client.get("/health")
    assert "X-Correlation-ID" in res1.headers
    assert len(res1.headers["X-Correlation-ID"]) > 10

    # Test client-provided correlation ID
    custom_cid = "custom-trace-uuid-12345"
    res2 = client.get("/health", headers={"X-Correlation-ID": custom_cid})
    assert res2.headers.get("X-Correlation-ID") == custom_cid


def test_payload_size_limit_guard():
    """Verify that payload exceeding maximum content length is rejected with 413."""
    # Send request with oversized content-length header
    fake_oversized_header = {"content-length": str(30 * 1024 * 1024)}  # 30MB
    response = client.post("/api/v1/chat", json={"message": "hello"}, headers=fake_oversized_header)
    assert response.status_code == 413
    data = response.json()
    assert "exceeds maximum permitted limit" in data.get("detail", "")


@pytest.mark.asyncio
async def test_database_health_check_probe():
    """Verify the database connectivity health check probe."""
    is_healthy = await check_database_health()
    assert isinstance(is_healthy, bool)


def test_docker_and_compose_configs_exist():
    """Verify Dockerfiles, .dockerignore, and docker-compose exist."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    assert (root_dir / "docker" / "Dockerfile.backend").exists()
    assert (root_dir / "docker" / "Dockerfile.frontend").exists()
    assert (root_dir / "docker-compose.yml").exists()
    assert (root_dir / ".dockerignore").exists()
