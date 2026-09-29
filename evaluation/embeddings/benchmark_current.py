"""Benchmark wrapper for Model A: Current Production Embedding Engine."""
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

proj_root = Path(__file__).resolve().parent.parent.parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from evaluation.embeddings.base import BaseEmbeddingBenchmarkModel
from rag.embeddings.encoder import MedicalEmbeddingEngine


class CurrentEmbeddingModel(BaseEmbeddingBenchmarkModel):
    """Model A: Current Production Medical Embedding Engine (128-dim hashing projection)."""

    def __init__(self, dimension: int = 128, device: str = "cpu"):
        super().__init__(
            model_name="current_production_embedding",
            dimension=dimension,
            max_seq_length=512,
            device=device
        )
        self.engine = MedicalEmbeddingEngine(embedding_dim=dimension)

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode clinical text excerpts using the current production encoder."""
        return self.engine.encode(texts)

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode user evaluation queries using the current production encoder."""
        return self.engine.encode(queries)

    def get_model_metadata(self) -> Dict[str, Any]:
        """Return deterministic model configuration."""
        return {
            "model_name": "Current Medical Dense Embedding Engine",
            "model_identifier": "current_production_v1",
            "model_revision": "1.0.0",
            "embedding_dimension": self.dimension,
            "tokenizer": "RegEx Tokenizer (word-boundary)",
            "max_sequence_length": self.max_seq_length,
            "pooling": "Token-hash Mean Pooling",
            "normalization": "L2 Normalization",
            "device": self.device,
            "license": "Project Internal / Apache-2.0",
            "library_versions": {
                "numpy": np.__version__
            }
        }
