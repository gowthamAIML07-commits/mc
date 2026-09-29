"""Integration tests verifying multi-task training pipelines execution and report outputs."""
import json
from pathlib import Path
import pytest
from training.pipelines.train_ocr import train_ocr_pipeline
from training.pipelines.train_augmentation import run_augmentation_pipeline
from training.pipelines.train_layout import train_layout_model
from training.pipelines.train_normalization import train_normalization_pipeline


REQUIRED_REPORT_FIELDS = [
    "pipeline_name",
    "model_id",
    "dataset_name",
    "dataset_version",
    "dataset_sha256",
    "training_seed",
    "training_time_seconds",
    "hardware_information",
    "model_configuration",
    "training_metrics",
    "validation_metrics",
    "test_metrics",
    "checkpoint_path"
]


def test_pipeline_a_ocr_execution():
    """Verify Pipeline A execution and report generation."""
    report = train_ocr_pipeline(seed=42)
    for field in REQUIRED_REPORT_FIELDS:
        assert field in report, f"OCR report missing required field: {field}"
    assert report["test_metrics"]["test_character_error_rate_cer"] >= 0.0
    assert report["test_metrics"]["test_field_extraction_accuracy"] > 0.80


def test_pipeline_c_augmentation_execution():
    """Verify Pipeline C synthetic generation execution."""
    report = run_augmentation_pipeline(count=25, seed=42)
    for field in REQUIRED_REPORT_FIELDS:
        assert field in report, f"Augmentation report missing required field: {field}"
    assert report["training_metrics"]["train_samples_generated"] >= 15


def test_pipeline_d_layout_execution():
    """Verify Pipeline D layout understanding execution."""
    report = train_layout_model(epochs=5, seed=42)
    for field in REQUIRED_REPORT_FIELDS:
        assert field in report, f"Layout report missing required field: {field}"
    assert report["test_metrics"]["key_value_pairing_accuracy"] >= 0.0


def test_pipeline_e_normalization_execution():
    """Verify Pipeline E medicine normalization execution."""
    report = train_normalization_pipeline(seed=42)
    for field in REQUIRED_REPORT_FIELDS:
        assert field in report, f"Normalization report missing required field: {field}"
    assert report["test_metrics"]["test_top1_normalization_accuracy"] >= 0.90
