"""Unit tests for the Configuration Loader and validation system."""
import os
from pathlib import Path
import pytest
from configs.loader import (
    AppConfig,
    DatasetsConfig,
    ModelsConfig,
    TrainingConfig,
    DatabaseConfig,
    RagConfig,
    expand_env_vars,
    load_yaml_file,
)


def test_expand_env_vars():
    """Verify environment variable expansion with default values."""
    data = {
        "host": "${TEST_HOST:-127.0.0.1}",
        "port": "${TEST_PORT:-8000}",
        "nested": {
            "name": "${TEST_NAME:-medicine_ai}"
        }
    }
    expanded = expand_env_vars(data)
    assert expanded["host"] == "127.0.0.1"
    assert expanded["port"] == "8000"
    assert expanded["nested"]["name"] == "medicine_ai"


def test_expand_env_vars_with_override(monkeypatch):
    """Verify environment variable expansion when env var is set."""
    monkeypatch.setenv("TEST_HOST", "192.168.1.100")
    data = {"host": "${TEST_HOST:-127.0.0.1}"}
    expanded = expand_env_vars(data)
    assert expanded["host"] == "192.168.1.100"


def test_load_all_configurations():
    """Verify that all platform configuration files load and validate properly."""
    app_config = AppConfig()
    
    # 1. Datasets Config
    assert isinstance(app_config.datasets, DatasetsConfig)
    assert app_config.datasets.registry_file == "data/registry/datasets.yaml"
    assert "ocr_document_extraction" in app_config.datasets.pipeline_dataset_mapping

    # 2. Models Config
    assert isinstance(app_config.models, ModelsConfig)
    assert app_config.models.runtime.device in ["auto", "cuda", "cpu"]
    assert "prescription_ocr" in app_config.models.models
    assert "safety_classifier" in app_config.models.models
    assert "llm_generation" in app_config.models.models

    # 3. Training Config
    assert isinstance(app_config.training, TrainingConfig)
    assert app_config.training.global_training.seed == 42
    assert "ocr_document_tuning" in app_config.training.pipelines
    assert "handwriting_recognition_training" in app_config.training.pipelines

    # 4. Database Config
    assert isinstance(app_config.database, DatabaseConfig)
    assert "users" in app_config.database.tables
    assert "prescriptions" in app_config.database.tables
    assert "drug_interactions" in app_config.database.tables

    # 5. RAG Config
    assert isinstance(app_config.rag, RagConfig)
    assert "drug_information" in app_config.rag.collections
    assert "medical_documents" in app_config.rag.collections
    assert app_config.rag.retrieval.get("hybrid_search") is True
