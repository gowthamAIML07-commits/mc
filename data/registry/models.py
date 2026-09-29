"""Dataset Registry and Licensing Data Models."""
from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, HttpUrl


class DatasetRole(str, Enum):
    TRAINING = "training"
    RETRIEVAL = "retrieval"
    EVALUATION = "evaluation"
    KNOWLEDGE_BASE = "knowledge_base"
    AUTOCOMPLETE = "autocomplete"
    AUGMENTATION = "augmentation"
    LAYOUT_PRETRAINING = "layout_pretraining"


class LicenseStatus(str, Enum):
    VERIFIED_PERMISSIVE = "verified_permissive" # MIT, Apache, CC BY, Public Domain
    VERIFIED_RESTRICTED = "verified_restricted" # UMLS / UTS agreement required
    VERIFIED_RESEARCH_ONLY = "verified_research_only" # Non-commercial research
    UNRESOLVED = "unresolved" # Prohibited from download/use until resolved


class SplitStrategy(BaseModel):
    train: float = Field(..., ge=0.0, le=1.0)
    val: float = Field(..., ge=0.0, le=1.0)
    test: float = Field(..., ge=0.0, le=1.0)
    seed: int = 42


class DatasetEntry(BaseModel):
    id: str = Field(..., description="Unique dataset identifier, e.g. DS-1")
    name: str = Field(..., description="Human-readable dataset name")
    task: str = Field(..., description="Primary clinical or vision pipeline task")
    role: Union[DatasetRole, str] = Field(..., description="Role: training, retrieval, evaluation, knowledge_base, etc.")
    source_url: str = Field(..., description="Official verifiable repository or authority URL")
    version: str = Field(..., description="Version or release identifier")
    license: str = Field(..., description="Formal license designation")
    redistribution_allowed: bool = Field(..., description="Whether redistribution is permitted")
    commercial_use_allowed: bool = Field(..., description="Whether commercial use is permitted")
    limitations: str = Field(..., description="Documented clinical, legal, or privacy limitations")
    local_path: str = Field(..., description="Designated local storage path")
    processed_path: str = Field(..., description="Path for cleaned/standardized artifacts")
    format: str = Field(..., description="File format specification")
    approximate_samples: Optional[int] = None
    split_strategy: Optional[SplitStrategy] = None
    manual_review_required: bool = False
    uts_license_required: bool = False


class DatasetRegistry(BaseModel):
    version: str
    registry_updated: str
    datasets: Dict[str, DatasetEntry]
