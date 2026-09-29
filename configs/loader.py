"""Configuration loader and schema validation engine."""
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


def expand_env_vars(data: Any) -> Any:
    """Recursively expand environment variables in strings: ${VAR:-default} or ${VAR}."""
    if isinstance(data, dict):
        return {k: expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [expand_env_vars(item) for item in data]
    elif isinstance(data, str):
        pattern = re.compile(r"\$\{([^}^{]+)\}")
        matches = pattern.findall(data)
        for match in matches:
            if ":-" in match:
                var_name, default_val = match.split(":-", 1)
            else:
                var_name, default_val = match, ""
            env_val = os.getenv(var_name, default_val)
            data = data.replace(f"${{{match}}}", env_val)
        return data
    return data


def load_yaml_file(file_path: Path) -> dict:
    """Load, parse, and expand environment variables in a YAML file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}
    return expand_env_vars(raw_data)


# ----------------------------------------------------------------------
# Typed Configuration Schemas
# ----------------------------------------------------------------------

class DatasetsConfig(BaseModel):
    registry_file: str
    data_directories: Dict[str, str]
    pipeline_dataset_mapping: Dict[str, Any]


class RuntimeConfig(BaseModel):
    device: str = "auto"
    mixed_precision: str = "fp16"
    num_threads: int = 4


class ModelsConfig(BaseModel):
    runtime: RuntimeConfig
    models: Dict[str, Any]


class GlobalTrainingConfig(BaseModel):
    seed: int = 42
    deterministic: bool = True
    logging_dir: str
    checkpoint_dir: str
    mlflow_tracking_uri: str
    experiment_name: str


class TrainingConfig(BaseModel):
    global_training: GlobalTrainingConfig
    pipelines: Dict[str, Any]


class DatabaseConfig(BaseModel):
    database: Dict[str, Any]
    tables: List[str]
    redis_cache: Dict[str, Any]


class RagConfig(BaseModel):
    qdrant: Dict[str, Any]
    collections: Dict[str, Any]
    ingestion: Dict[str, Any]
    retrieval: Dict[str, Any]


class AppConfig:
    """Consolidated configuration manager for Medicine AI."""

    def __init__(self, configs_dir: Optional[Path] = None):
        if configs_dir is None:
            self.configs_dir = Path(__file__).resolve().parent
        else:
            self.configs_dir = Path(configs_dir)

        self.datasets: DatasetsConfig = self._load_datasets_config()
        self.models: ModelsConfig = self._load_models_config()
        self.training: TrainingConfig = self._load_training_config()
        self.database: DatabaseConfig = self._load_database_config()
        self.rag: RagConfig = self._load_rag_config()

    def _load_datasets_config(self) -> DatasetsConfig:
        data = load_yaml_file(self.configs_dir / "datasets.yaml")
        return DatasetsConfig(**data)

    def _load_models_config(self) -> ModelsConfig:
        data = load_yaml_file(self.configs_dir / "models.yaml")
        return ModelsConfig(**data)

    def _load_training_config(self) -> TrainingConfig:
        data = load_yaml_file(self.configs_dir / "training.yaml")
        return TrainingConfig(**data)

    def _load_database_config(self) -> DatabaseConfig:
        data = load_yaml_file(self.configs_dir / "database.yaml")
        return DatabaseConfig(**data)

    def _load_rag_config(self) -> RagConfig:
        data = load_yaml_file(self.configs_dir / "rag.yaml")
        return RagConfig(**data)


# Global singleton instance
config = AppConfig()
