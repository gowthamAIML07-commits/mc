"""Unit tests for synthetic prescription generator, deterministic isolation, and schema."""
import json
from pathlib import Path
import pytest
from ml.ocr.augmentation import SyntheticPrescriptionGenerator


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def test_synthetic_generator_deterministic_generation(tmp_path: Path):
    gen1 = SyntheticPrescriptionGenerator(output_dir=tmp_path / "run1")
    img1, ann1 = gen1.generate_single_prescription(index=0, split="train", seed=42)

    gen2 = SyntheticPrescriptionGenerator(output_dir=tmp_path / "run2")
    img2, ann2 = gen2.generate_single_prescription(index=0, split="train", seed=42)

    assert ann1["prescription_id"] == ann2["prescription_id"]
    assert ann1["date"] == ann2["date"]
    assert ann1["medicines"] == ann2["medicines"]
    assert ann1["disclaimer"] == ann2["disclaimer"]
    assert img1.size == img2.size
    assert img1.tobytes() == img2.tobytes()



def test_synthetic_generator_split_isolation(tmp_path: Path):
    gen = SyntheticPrescriptionGenerator(output_dir=tmp_path)
    img_train, ann_train = gen.generate_single_prescription(index=0, split="train", seed=42)
    img_test, ann_test = gen.generate_single_prescription(index=0, split="test", seed=42)

    # Train and test sample 0 should have different seeds and content
    assert ann_train["seed_used"] != ann_test["seed_used"]
    assert ann_train["split"] == "train"
    assert ann_test["split"] == "test"


def test_synthetic_watermark_and_zero_phi(tmp_path: Path, project_root: Path):
    gen = SyntheticPrescriptionGenerator(output_dir=tmp_path)
    img, ann = gen.generate_single_prescription(index=1, split="train", seed=100)

    # Must contain explicit synthetic disclaimer
    assert "SYNTHETIC SPECIMEN" in ann["disclaimer"]
    assert ann["is_synthetic"] is True

    # Check zero real PHI
    assert "Synthetic" in ann["patient"]["name"]
    assert "Synthetic" in ann["doctor"]["doctor"]

    # Verify medicine names originate from approved vocabulary
    approved_vocab_file = project_root / "data" / "processed" / "normalization" / "approved_vocabulary.json"
    if approved_vocab_file.exists():
        with open(approved_vocab_file, "r", encoding="utf-8") as f:
            valid_vocab = set(json.load(f))
        for med in ann["medicines"]:
            assert med["canonical_name"] in valid_vocab or med["ingredient"] in valid_vocab
