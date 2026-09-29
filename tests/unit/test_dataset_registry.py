"""Unit tests for the Dataset Registry manifest and metadata validation."""
import pytest
from data.registry.verifier import DatasetLicenseVerifier
from data.registry.models import DatasetRole


@pytest.fixture
def verifier():
    return DatasetLicenseVerifier()


def test_registry_loaded_successfully(verifier):
    """Verify registry parses and contains all 6 required datasets."""
    assert verifier.registry is not None
    assert len(verifier.registry.datasets) == 6
    expected_keys = [
        "indian_medical_prescription_ocr",
        "doctors_handwritten_prescription_bd",
        "rxnorm",
        "rxterms",
        "funsd",
        "synthetic_prescriptions"
    ]
    for key in expected_keys:
        assert key in verifier.registry.datasets, f"Missing expected dataset key: {key}"


def test_dataset_required_metadata_fields(verifier):
    """Verify that every dataset has all 7 mandatory metadata fields."""
    for key, ds in verifier.registry.datasets.items():
        assert ds.id.startswith("DS-"), f"Dataset {key} ID must start with DS-"
        assert ds.name, f"Dataset {key} missing name"
        assert ds.source_url.startswith("http") or ds.source_url.startswith("internal_generator"), (
            f"Dataset {key} has invalid source URL: {ds.source_url}"
        )
        assert ds.license, f"Dataset {key} missing license"
        assert ds.version, f"Dataset {key} missing version"
        assert ds.task, f"Dataset {key} missing intended task"
        assert ds.role, f"Dataset {key} missing role"
        assert ds.limitations, f"Dataset {key} missing limitations"


def test_dataset_roles_correctly_assigned(verifier):
    """Verify dataset roles map precisely to intended multi-task ML architecture."""
    indian_rx = verifier.get_dataset("DS-1")
    assert indian_rx.role == "training"
    assert "ocr" in indian_rx.task

    bd_handwritten = verifier.get_dataset("DS-2")
    assert bd_handwritten.role == "training"
    assert "handwritten" in bd_handwritten.task

    rxnorm = verifier.get_dataset("DS-3")
    assert rxnorm.role == "knowledge_base"
    assert "normalization" in rxnorm.task

    rxterms = verifier.get_dataset("DS-4")
    assert rxterms.role == "autocomplete"
    assert "autocomplete" in rxterms.task

    funsd = verifier.get_dataset("DS-5")
    assert funsd.role == "layout_pretraining"

    synthetic = verifier.get_dataset("DS-6")
    assert synthetic.role == "augmentation"


def test_dataset_lookup_by_id_and_key(verifier):
    """Verify flexible lookup by key or by DS-ID."""
    ds_by_key = verifier.get_dataset("rxnorm")
    ds_by_id = verifier.get_dataset("DS-3")
    assert ds_by_key is not None
    assert ds_by_id is not None
    assert ds_by_key.id == ds_by_id.id
