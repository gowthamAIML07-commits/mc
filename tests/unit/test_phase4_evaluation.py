"""Unit and integration tests for Phase 4 multi-task training outputs and schemas."""
import json
from pathlib import Path
import pytest
import torch


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def test_phase4_training_config_valid(project_root: Path):
    cfg_path = project_root / "configs" / "training_phase4.yaml"
    assert cfg_path.exists(), f"Configuration file missing: {cfg_path}"

    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "reproducibility" in cfg
    assert cfg["reproducibility"]["global_seed"] == 42
    assert "pipelines" in cfg
    assert "handwriting_recognition" in cfg["pipelines"]
    assert "prescription_ocr" in cfg["pipelines"]
    assert "layout_parsing" in cfg["pipelines"]
    assert "medicine_normalization" in cfg["pipelines"]


def test_phase4_handwriting_checkpoint_schema(project_root: Path):
    ckpt_path = project_root / "checkpoints" / "phase4" / "handwriting" / "best_handwriting_crnn.pt"
    if ckpt_path.exists():
        data = torch.load(ckpt_path, map_location="cpu")
        assert "model_state_dict" in data
        assert "label_map" in data
        assert len(data["label_map"]) == 78
        assert "best_val_top1_acc" in data
        assert "seed" in data


def test_phase4_collision_hash_preservation(project_root: Path):
    manifest_path = project_root / "data" / "manifests" / "handwriting_dataset.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 55 cross-split collision hashes must be recorded
    assert manifest["cross_split_duplicates_count"] == 55
    assert len(manifest["cross_split_duplicates"]) > 0

    # Splits must not have been modified
    splits = manifest["splits"]
    assert splits["training"]["sample_count"] == 3120
    assert splits["validation"]["sample_count"] == 780
    assert splits["testing"]["sample_count"] == 780


def test_phase4_end_to_end_schema(project_root: Path):
    from ml.ocr.pipeline import PrescriptionOCRExtractor
    from ml.embeddings.normalizer import MedicineNormalizer

    extractor = PrescriptionOCRExtractor()
    normalizer = MedicineNormalizer()

    sample_rx = "<s_ocr> doctor_name: Dr. Test clinic_name: Clinic A patient_name: Patient X medications: Tab. Dolo 650mg </s_ocr>"
    extracted = extractor.extract_from_text(sample_rx)
    assert extracted.doctor_name == "Dr. Test"
    assert len(extracted.medications) >= 1

    norm = normalizer.normalize(extracted.medications[0].name)
    assert norm["rxnorm_id"] == "161"
    assert norm["verification_status"] == "verified"
