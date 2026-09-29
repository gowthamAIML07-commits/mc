"""Unit tests for Doctor's Handwritten Prescription BD dataset manifest and integrity."""
import json
from pathlib import Path
import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def test_handwriting_manifest_schema_and_counts(project_root: Path):
    manifest_path = project_root / "data" / "manifests" / "handwriting_dataset.json"
    assert manifest_path.exists(), f"Manifest not found: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset_name"] == "Doctor's Handwritten Prescription BD Dataset"
    assert data["total_samples"] == 4680
    assert data["total_classes"] == 78
    assert data["total_generics"] == 15
    assert len(data["classes_list"]) == 78
    assert "splits" in data
    assert set(data["splits"].keys()) == {"training", "validation", "testing"}

    splits = data["splits"]
    assert splits["training"]["sample_count"] == 3120
    assert splits["validation"]["sample_count"] == 780
    assert splits["testing"]["sample_count"] == 780


def test_handwriting_image_health_and_integrity(project_root: Path):
    manifest_path = project_root / "data" / "manifests" / "handwriting_dataset.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    splits = data["splits"]
    for split_name, sdata in splits.items():
        assert sdata["missing_images"] == 0, f"Found missing images in {split_name}"
        assert sdata["corrupt_images"] == 0, f"Found corrupt images in {split_name}"
        assert sdata["images_verified"] == sdata["sample_count"]
        assert sdata["unique_medicines"] == 78
        assert sdata["unique_generics"] == 15
        assert len(sdata["representative_samples"]) > 0


def test_handwriting_duplicates_and_class_mapping(project_root: Path):
    manifest_path = project_root / "data" / "manifests" / "handwriting_dataset.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "cross_split_duplicates_count" in data
    assert isinstance(data["cross_split_duplicates_count"], int)

    # Verify class mapping file
    mapping_path = project_root / "data" / "processed" / "bd_handwritten" / "class_mapping.json"
    assert mapping_path.exists(), f"Class mapping not found: {mapping_path}"

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    assert len(mapping["classes"]) == 78
    assert len(mapping["class_to_idx"]) == 78
    assert len(mapping["idx_to_class"]) == 78


def test_handwriting_markdown_report_exists(project_root: Path):
    report_path = project_root / "docs" / "datasets" / "handwriting_dataset_report.md"
    assert report_path.exists(), f"Report not found: {report_path}"

    content = report_path.read_text(encoding="utf-8")
    assert "Doctor's Handwritten Prescription BD Dataset" in content
    assert "Split Analysis & Partition Statistics" in content
    assert "Duplicate & Cross-Split Leakage Analysis" in content
