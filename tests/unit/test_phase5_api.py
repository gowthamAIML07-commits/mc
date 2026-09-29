"""Unit tests for Phase 5 FastAPI endpoints, upload validation, and ModelRegistry."""
import io
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from apps.backend.core.config import settings
from apps.backend.main import app
from apps.backend.services.model_registry import ModelRegistry, ModelRegistryError
from apps.backend.services.upload_validator import ImageUploadValidator, UploadValidationError
from rag.retrieval.base import InMemoryRxNormRetriever


@pytest.fixture
def client() -> TestClient:
    # Initialize model registry for test client
    registry = ModelRegistry.get_instance()
    if not registry.is_ready():
        registry.load_all_models()
    return TestClient(app)


def create_test_image_bytes(format_name: str = "PNG", size: tuple = (200, 200), color: tuple = (255, 255, 255)) -> bytes:
    """Helper to generate in-memory valid test image bytes."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


# --- Endpoint Tests: Health & Readiness ---

def test_health_endpoint(client: TestClient):
    """Test 24: Health liveness endpoint."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == settings.APP_NAME


def test_readiness_endpoint(client: TestClient):
    """Test 25: Readiness probe checking loaded Phase 4 models."""
    res = client.get("/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["models_loaded"] is True
    assert "metadata" in data


# --- Upload Validation Tests ---

def test_valid_png_upload(client: TestClient):
    """Test 2: Valid PNG prescription upload."""
    img_bytes = create_test_image_bytes("PNG")
    response = client.post(
        "/api/v1/prescriptions/upload-and-extract",
        files={"file": ("test_prescription.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert "data" in payload
    assert payload["data"]["prescription_id"].startswith("rx_")


def test_valid_jpeg_upload(client: TestClient):
    """Test 1: Valid JPEG prescription upload."""
    img_bytes = create_test_image_bytes("JPEG")
    response = client.post(
        "/api/v1/prescriptions/upload-and-extract",
        files={"file": ("test_prescription.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True


def test_valid_webp_upload(client: TestClient):
    """Test 3: Valid WEBP prescription upload."""
    img_bytes = create_test_image_bytes("WEBP")
    response = client.post(
        "/api/v1/prescriptions/upload-and-extract",
        files={"file": ("test_prescription.webp", img_bytes, "image/webp")}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True


def test_missing_file_upload(client: TestClient):
    """Test 4: Missing file payload returns 422/400."""
    response = client.post("/api/v1/prescriptions/upload-and-extract")
    assert response.status_code in [400, 422]


def test_unsupported_file_extension(client: TestClient):
    """Test 5: Unsupported file type rejected."""
    response = client.post(
        "/api/v1/prescriptions/upload-and-extract",
        files={"file": ("prescription.pdf", b"%PDF-1.4 dummy", "application/pdf")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error_code"] == "UNSUPPORTED_EXTENSION"


def test_fake_image_content(client: TestClient):
    """Test 6: Text file renamed to .png rejected by image verifier."""
    fake_bytes = b"This is just plain text, not an actual image binary stream."
    response = client.post(
        "/api/v1/prescriptions/upload-and-extract",
        files={"file": ("fake.png", fake_bytes, "image/png")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error_code"] in ["CORRUPTED_IMAGE", "DECODING_FAILED"]


def test_corrupted_image_header(client: TestClient):
    """Test 7: Corrupted image stream."""
    corrupt_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 20
    response = client.post(
        "/api/v1/prescriptions/upload-and-extract",
        files={"file": ("corrupt.png", corrupt_bytes, "image/png")}
    )
    assert response.status_code == 400
    assert response.json()["error_code"] in ["CORRUPTED_IMAGE", "DECODING_FAILED"]


def test_oversized_file_rejected():
    """Test 8: File exceeding MAX_UPLOAD_SIZE_MB rejected."""
    large_bytes = b"0" * (settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1024)
    with pytest.raises(UploadValidationError) as exc:
        ImageUploadValidator.validate_image_bytes(large_bytes, "large.png")
    assert exc.value.error_code == "FILE_TOO_LARGE"


def test_invalid_image_dimensions():
    """Test 9: Dimensions too small or too large rejected."""
    # Too small (10x10 px)
    tiny_bytes = create_test_image_bytes("PNG", size=(10, 10))
    with pytest.raises(UploadValidationError) as exc:
        ImageUploadValidator.validate_image_bytes(tiny_bytes, "tiny.png")
    assert exc.value.error_code == "IMAGE_TOO_SMALL"


# --- ModelRegistry & Architecture Tests ---

def test_model_registry_initialization():
    """Test 10 & 13: ModelRegistry initializes on CPU and verifies components."""
    registry = ModelRegistry.get_instance()
    registry.load_all_models()
    assert registry.is_ready() is True
    assert registry.handwriting_model is not None
    assert registry.ocr_extractor is not None
    assert registry.layout_parser is not None
    assert registry.normalizer is not None


def test_missing_checkpoint_raises_error(tmp_path: Path):
    """Test 11: Missing checkpoint path raises ModelRegistryError."""
    registry = ModelRegistry()
    non_existent = tmp_path / "non_existent_crnn.pt"
    with pytest.raises(ModelRegistryError) as exc:
        registry.load_all_models(handwriting_path=str(non_existent))
    assert "not found" in str(exc.value)


def test_incompatible_checkpoint_raises_error(tmp_path: Path):
    """Test 12: Corrupted or incompatible checkpoint raises ModelRegistryError."""
    corrupt_ckpt = tmp_path / "bad_checkpoint.pt"
    corrupt_ckpt.write_bytes(b"Invalid pytorch bytes")
    registry = ModelRegistry()
    with pytest.raises(ModelRegistryError):
        registry.load_all_models(handwriting_path=str(corrupt_ckpt))


# --- RAG Interface Preparation Tests ---

@pytest.mark.asyncio
async def test_in_memory_rxnorm_retriever():
    """Test RAG retrieval boundary on verified RxNorm monographs."""
    retriever = InMemoryRxNormRetriever()
    results = await retriever.search(query="Paracetamol", top_k=3)
    assert len(results) > 0
    top_doc = results[0]
    assert "Paracetamol" in top_doc.text or "Acetaminophen" in top_doc.text
    assert top_doc.score > 0.50
    assert top_doc.metadata["rxcui"] == "161"
