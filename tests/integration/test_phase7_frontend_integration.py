"""Integration Tests for Phase 7 Frontend Serving and Client-Backend Integration."""
import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_serve_frontend_root():
    """Verify GET / successfully serves the clinical workspace HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AI MEDICINE ASSISTANT" in response.text
    assert "Prescription Scanner" in response.text
    assert "Clinical Chat" in response.text


def test_serve_frontend_app_route():
    """Verify GET /app successfully serves the clinical workspace HTML."""
    response = client.get("/app")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Prescription Scanner" in response.text


def test_serve_static_assets():
    """Verify static styles and scripts are served with correct MIME types."""
    css_res = client.get("/static/styles.css")
    assert css_res.status_code == 200
    assert "text/css" in css_res.headers["content-type"]
    assert "--bg-app:" in css_res.text

    app_js_res = client.get("/static/app.js")
    assert app_js_res.status_code == 200
    assert "javascript" in app_js_res.headers["content-type"]

    api_js_res = client.get("/static/api.js")
    assert api_js_res.status_code == 200
    assert "javascript" in api_js_res.headers["content-type"]


def test_frontend_chat_end_to_end_flow():
    """Test full frontend-structured API flow: Chat request with prescription context."""
    chat_payload = {
        "message": "Can I take Aspirin with my prescription?",
        "conversation_id": "fe_test_conv_01",
        "prescription_context": {
            "candidates": [
                {
                    "raw_text": "Warfarin 5mg",
                    "normalized_name": "Warfarin Sodium 5 MG Oral Tablet",
                    "verification_status": "verified",
                    "confidence": 0.98
                }
            ]
        }
    }
    response = client.post("/api/v1/chat", json=chat_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "DRUG_INTERACTION"
    assert len(data["interactions"]) >= 1
    assert data["interactions"][0]["severity"] == "MAJOR"
    assert "bleeding" in data["interactions"][0]["clinical_effect"].lower()
