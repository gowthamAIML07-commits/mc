"""Base class and common utilities for medical embedding benchmark models."""
import abc
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("medicine_ai.eval.embeddings")


class BaseEmbeddingBenchmarkModel(abc.ABC):
    """Abstract interface for benchmark embedding models with built-in sanity checks."""

    def __init__(self, model_name: str, dimension: int, max_seq_length: int = 512, device: str = "cpu"):
        self.model_name = model_name
        self.dimension = dimension
        self.max_seq_length = max_seq_length
        self.device = device

    @abc.abstractmethod
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode list of clinical texts into L2-normalized 2D numpy array [N, D]."""
        pass

    @abc.abstractmethod
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode evaluation queries into L2-normalized 2D numpy array [Q, D]."""
        pass

    @abc.abstractmethod
    def get_model_metadata(self) -> Dict[str, Any]:
        """Return deterministic configuration parameters."""
        pass

    def run_sanity_checks(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Verify dimensional shape, NaN/Inf absence, non-zero energy, and unit L2 norms."""
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")
        
        n_samples, dim = embeddings.shape
        if dim != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {dim}")

        has_nan = bool(np.isnan(embeddings).any())
        has_inf = bool(np.isinf(embeddings).any())
        
        norms = np.linalg.norm(embeddings, axis=1)
        zero_vectors = int((norms < 1e-6).sum())
        mean_norm = float(np.mean(norms))
        min_norm = float(np.min(norms))
        max_norm = float(np.max(norms))

        is_normalized = (abs(mean_norm - 1.0) < 1e-3) or (zero_vectors == n_samples)

        checks = {
            "n_samples": n_samples,
            "dimension": dim,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "zero_vector_count": zero_vectors,
            "mean_l2_norm": round(mean_norm, 5),
            "min_l2_norm": round(min_norm, 5),
            "max_l2_norm": round(max_norm, 5),
            "is_l2_normalized": is_normalized,
            "sanity_passed": (not has_nan and not has_inf and zero_vectors == 0 and is_normalized)
        }
        return checks
