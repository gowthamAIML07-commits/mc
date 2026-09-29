"""Integration Tests for Phase 8 End-to-End Deployment Readiness."""
import pytest
from scripts.verify_deployment import run_deployment_verification


@pytest.mark.asyncio
async def test_end_to_end_deployment_readiness_verification():
    """Verify that all subsystems pass deployment readiness probe."""
    is_ready = await run_deployment_verification()
    assert is_ready is True
