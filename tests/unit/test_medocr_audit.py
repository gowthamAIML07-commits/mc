"""Unit tests for the MedOCR-Vision dataset audit manifest and documentation."""

import json
from pathlib import Path
import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def test_audit_manifest_exists_and_valid(project_root: Path):
    manifest_path = project_root / "data" / "manifests" / "medocr_vision_audit.json"
    assert manifest_path.exists(), f"Audit manifest not found at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset_name"] == "MedOCR-Vision Dataset (Indian Medical Prescription OCR)"
    assert data["total_samples"] == 2462
    assert "splits" in data
    assert set(data["splits"].keys()) == {"train", "validation", "test"}


def test_audit_split_sample_counts(project_root: Path):
    manifest_path = project_root / "data" / "manifests" / "medocr_vision_audit.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    splits = data["splits"]
    assert splits["train"]["sample_count"] == 1969
    assert splits["validation"]["sample_count"] == 246
    assert splits["test"]["sample_count"] == 247

    for split_name, split_data in splits.items():
        assert split_data["missing_images"] == 0, f"Missing images in {split_name}"
        assert split_data["corrupt_images"] == 0, f"Corrupt images in {split_name}"
        assert split_data["text_statistics"]["empty_text_count"] == 0, f"Empty text in {split_name}"
        assert len(split_data["representative_samples"]) > 0, f"No samples in {split_name}"


def test_audit_category_and_leakage_findings(project_root: Path):
    manifest_path = project_root / "data" / "manifests" / "medocr_vision_audit.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cat_dist = data["overall_category_distribution"]
    assert "prescription" in cat_dist
    assert "general_ocr" in cat_dist
    assert "medical_document" in cat_dist
    # Verify that general_ocr contamination is recognized (> 500 samples)
    assert cat_dist["general_ocr"] > 500, "General OCR receipt contamination should be tracked"

    # Verify leakage tracking
    leakage = data["cross_split_leakage"]
    assert "exact_image_overlap" in leakage
    assert "exact_text_overlap" in leakage


def test_audit_markdown_report_exists(project_root: Path):
    doc_path = project_root / "docs" / "datasets" / "medocr_vision_audit.md"
    assert doc_path.exists(), f"Markdown audit report not found at {doc_path}"

    content = doc_path.read_text(encoding="utf-8")
    assert "Executive Summary" in content
    assert "Split-by-Split Descriptive Statistics" in content
    assert "Content Domain & Semantic Breakdown" in content
    assert "Data Leakage & Duplication Findings" in content
    assert "License, Provenance, & Legal Risk Analysis" in content
    assert "Suitability Assessment Across 5 OCR Tasks" in content
    assert "Comparative Dataset Role Mapping" in content
