"""Unit tests for RxNorm / RxTerms clinical normalization and multi-tier matching."""
import json
from pathlib import Path
import pytest
from ml.embeddings.normalizer import MedicineNormalizer


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def normalizer() -> MedicineNormalizer:
    return MedicineNormalizer()


def test_normalization_manifest_and_provenance(project_root: Path):
    manifest_path = project_root / "data" / "manifests" / "rxnorm_rxterms_manifest.json"
    assert manifest_path.exists(), f"Manifest not found: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "RxNorm" in data["resource_name"]
    assert data["provenance_tracked"] is True
    assert data["concept_count"] >= 15
    assert len(data["matching_tiers"]) >= 4

    vocab_path = project_root / "data" / "processed" / "normalization" / "approved_vocabulary.json"
    assert vocab_path.exists(), f"Approved vocabulary not found: {vocab_path}"


def test_exact_and_alias_normalization(normalizer: MedicineNormalizer):
    # Test canonical exact match
    res1 = normalizer.normalize("Amoxicillin 500 MG Oral Tablet")
    assert res1["rxnorm_id"] == "308189"
    assert res1["verification_status"] == "verified"
    assert res1["match_tier"] in ["exact_hash", "exact_sanitized"]

    # Test trade alias match
    res2 = normalizer.normalize("Dolo 650")
    assert res2["rxnorm_id"] == "161"
    assert res2["ingredient"] == "Paracetamol"
    assert res2["verification_status"] == "verified"

    res3 = normalizer.normalize("Pan 40")
    assert res3["rxnorm_id"] == "312615"
    assert res3["ingredient"] == "Pantoprazole"
    assert res3["verification_status"] == "verified"


def test_clinical_abbreviation_resolution(normalizer: MedicineNormalizer):
    res_pcm = normalizer.normalize("PCM")
    assert res_pcm["rxnorm_id"] == "161"
    assert res_pcm["ingredient"] == "Paracetamol"

    res_amx = normalizer.normalize("AMX")
    assert res_amx["rxnorm_id"] == "308189"
    assert res_amx["ingredient"] == "Amoxicillin"

    res_azm = normalizer.normalize("AZM")
    assert res_azm["rxnorm_id"] == "198440"
    assert res_azm["ingredient"] == "Azithromycin"


def test_phonetic_and_fuzzy_normalization(normalizer: MedicineNormalizer):
    # Phonetic variations with spelling noise
    res_phonetic1 = normalizer.normalize("Amoxcillin 500mg")
    assert res_phonetic1["rxnorm_id"] == "308189"
    assert res_phonetic1["confidence"] >= 0.80

    res_phonetic2 = normalizer.normalize("Paractaml 650")
    assert res_phonetic2["rxnorm_id"] == "161"
    assert res_phonetic2["confidence"] >= 0.80

    res_phonetic3 = normalizer.normalize("Metfornin 500")
    assert res_phonetic3["rxnorm_id"] == "860975"
    assert res_phonetic3["confidence"] >= 0.80

    res_phonetic4 = normalizer.normalize("Cetrizine 10")
    assert res_phonetic4["rxnorm_id"] == "310489"
    assert res_phonetic4["confidence"] >= 0.80


def test_unrecognized_entity_handling(normalizer: MedicineNormalizer):
    res_invalid = normalizer.normalize("CompletelyUnknownChemicalXYZ999")
    assert res_invalid["rxnorm_id"] is None
    assert res_invalid["verification_status"] == "unverified"
    assert res_invalid["confidence"] == 0.0
