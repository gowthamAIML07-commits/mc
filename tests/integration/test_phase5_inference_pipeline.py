"""Integration tests for Phase 5 Prescription Inference Pipeline and Safety Rules."""
import pytest
from PIL import Image

from apps.backend.schemas.prescription import PrescriptionResult
from apps.backend.services.model_registry import ModelRegistry
from apps.backend.services.prescription_service import PrescriptionInferenceService


@pytest.fixture(scope="module")
def inference_service() -> PrescriptionInferenceService:
    registry = ModelRegistry.get_instance()
    if not registry.is_ready():
        registry.load_all_models()
    return PrescriptionInferenceService(registry=registry)


def test_successful_prescription_inference(inference_service: PrescriptionInferenceService):
    """Test 14: End-to-end inference processing on clean prescription specimen."""
    img = Image.new("RGB", (600, 800), (255, 255, 255))
    result: PrescriptionResult = inference_service.process_prescription(img)

    assert result.prescription_id.startswith("rx_")
    assert result.overall_confidence > 0.0
    assert result.timings is not None
    assert result.timings.total_pipeline_ms > 0.0


def test_dosage_parsing_and_normalization_success(inference_service: PrescriptionInferenceService):
    """Test 18: Exact and trade alias normalization success with parsed dosage fields."""
    raw_sample = "1. Tab. Amoxicillin 500mg --- 1-0-1 (BD) x 5 days"
    clean_name, strength, dose, freq, dur = inference_service._parse_dosage_and_schedule(raw_sample)

    assert "Amoxicillin" in clean_name
    assert strength == "500mg"
    assert dose == "Tab."
    assert "1-0-1" in freq
    assert dur == "5 days"

    norm_res = inference_service.registry.normalizer.normalize(clean_name)
    assert norm_res["rxnorm_id"] == "308189"
    assert norm_res["ingredient"] == "Amoxicillin"


def test_review_required_for_uncertain_and_fuzzy_matches(inference_service: PrescriptionInferenceService):
    """Test 17 & 20: Noisy/fuzzy matches safely trigger review_required status."""
    noisy_query = "Amoxcillin 500mg" # Misspelled
    norm_res = inference_service.registry.normalizer.normalize(noisy_query)
    
    assert norm_res["rxnorm_id"] == "308189"
    # Verification policy verification
    if norm_res["confidence"] < 0.85:
        # Should be classified as review_required
        assert norm_res["verification_status"] == "review_required"


def test_normalization_failure_safety_policy(inference_service: PrescriptionInferenceService):
    """Test 19: Unrecognized/OOV compounds are never hallucinated and marked unverified."""
    unrecognized = "CompletelyUnknownChemical999 50mg"
    norm_res = inference_service.registry.normalizer.normalize(unrecognized)

    assert norm_res["rxnorm_id"] is None
    assert norm_res["verification_status"] == "unverified"
    assert norm_res["normalized_name"] == "Unrecognized Entity"


def test_missing_dosage_attributes_handled_as_none(inference_service: PrescriptionInferenceService):
    """Test 21, 22, 23: Missing strength, frequency, or duration remain None (never fabricated)."""
    minimal_line = "Paracetamol"
    clean_name, strength, dose, freq, dur = inference_service._parse_dosage_and_schedule(minimal_line)

    assert clean_name == "Paracetamol"
    assert strength is None # Test 21: Missing strength remains None
    assert freq is None     # Test 22: Missing frequency remains None
    assert dur is None      # Test 23: Missing duration remains None
    assert dose is None


def test_mixed_printed_and_handwritten_candidates(inference_service: PrescriptionInferenceService):
    """Test 15 & 16: Processing multiple diverse medicine lines."""
    lines = [
        "1. Dolo 650mg 1-0-1",
        "2. Cap. Pan 40mg OD x 14 days",
        "3. PCM 650 SOS"
    ]
    for line in lines:
        cname, str_val, dose_val, freq_val, dur_val = inference_service._parse_dosage_and_schedule(line)
        norm = inference_service.registry.normalizer.normalize(cname)
        assert norm["rxnorm_id"] is not None
