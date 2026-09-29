"""Integration Tests for Phase 6 Chat API Endpoint and End-to-End Clinical Pipeline."""
import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_chat_api_medicine_information_query():
    """Test standard clinical inquiry over knowledge base."""
    payload = {
        "message": "What is Metformin used for and what are its common indications?",
        "conversation_id": "test_conv_001"
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == "test_conv_001"
    assert data["safety_level"] in ["LOW", "MODERATE"]
    assert len(data["evidence"]) >= 1
    assert len(data["citations"]) >= 1
    assert "Metformin" in data["response"]
    assert "type 2 diabetes" in data["response"].lower() or "glycemic" in data["response"].lower()


def test_chat_api_drug_interaction_query():
    """Test structured drug-drug interaction detection."""
    payload = {
        "message": "Can I take Warfarin together with Aspirin?",
        "conversation_id": "test_conv_002"
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "DRUG_INTERACTION"
    assert len(data["interactions"]) >= 1
    inter = data["interactions"][0]
    assert inter["interaction_found"] is True
    assert inter["severity"] == "MAJOR"
    assert "bleeding" in inter["clinical_effect"].lower()
    assert data["requires_professional_review"] is True


def test_chat_api_emergency_short_circuit():
    """Test that emergency queries are immediately routed to emergency safety guidance."""
    payload = {
        "message": "I have severe crushing chest pain, difficulty breathing, and my arm is numb.",
        "conversation_id": "test_conv_003"
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["safety_level"] == "EMERGENCY"
    assert data["intent"] == "EMERGENCY"
    assert data["requires_professional_review"] is True
    assert "CRITICAL MEDICAL ALERT" in data["response"]
    assert "emergency services" in data["response"].lower() or "911" in data["response"]


def test_chat_api_empty_message_rejected():
    """Test that empty or whitespace-only messages are rejected with 422."""
    response = client.post("/api/v1/chat", json={"message": "   "})
    assert response.status_code == 422


def test_chat_api_prescription_context_verified():
    """Test consuming verified prescription context from Phase 5."""
    payload = {
        "message": "Please explain the medications in my prescription.",
        "conversation_id": "test_conv_004",
        "prescription_context": {
            "candidates": [
                {
                    "raw_text": "Amoxicillin 500mg",
                    "normalized_name": "Amoxicillin",
                    "verification_status": "verified",
                    "confidence": 0.98
                }
            ]
        }
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "Prescription Context (Verified)" in data["response"]
    assert "Amoxicillin" in data["response"]


def test_chat_api_prescription_context_unverified_requires_review():
    """Test that unverified prescription handwriting triggers explicit safety warning."""
    payload = {
        "message": "What should I take from my prescription?",
        "conversation_id": "test_conv_005",
        "prescription_context": {
            "candidates": [
                {
                    "raw_text": "met...",
                    "normalized_name": None,
                    "verification_status": "review_required",
                    "confidence": 0.35
                }
            ]
        }
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["requires_professional_review"] is True
    assert "Prescription Warning" in data["response"]
    assert "could not be reliably verified" in data["response"]
    assert any("requires pharmacist inspection" in w for w in data["warnings"])


def test_chat_api_out_of_scope_query():
    """Test non-medical inquiries receive appropriate boundary guidance."""
    payload = {
        "message": "How do I write a Python function to sort a list?",
        "conversation_id": "test_conv_006"
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["safety_level"] == "OUT_OF_SCOPE"
    assert "AI Clinical Medicine Assistant" in data["response"]
