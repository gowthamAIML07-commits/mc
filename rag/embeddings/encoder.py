"""Deterministic Dense Embedding Engine for Medical Knowledge Representation."""
import hashlib
import re
from typing import List, Union
import numpy as np
import torch


class MedicalEmbeddingEngine:
    """Computes normalized dense embeddings for medical texts and queries."""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim

    def _hash_token_to_embedding(self, token: str) -> np.ndarray:
        """Map text token to deterministic pseudo-dense vector space."""
        h = hashlib.sha256(token.encode("utf-8")).digest()
        # Derive deterministic float vector from hash bytes
        vals = [((b / 255.0) * 2.0 - 1.0) for b in h[:self.embedding_dim]]
        if len(vals) < self.embedding_dim:
            vals.extend([0.0] * (self.embedding_dim - len(vals)))
        return np.array(vals[:self.embedding_dim], dtype=np.float32)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode single text or list of texts into L2-normalized dense embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            tokens = re.findall(r"\b\w+\b", text.lower())
            if not tokens:
                vec = np.zeros(self.embedding_dim, dtype=np.float32)
            else:
                token_vecs = [self._hash_token_to_embedding(tok) for tok in tokens]
                vec = np.mean(token_vecs, axis=0)

            # L2 Normalize
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec = vec / norm
            embeddings.append(vec)

        return np.array(embeddings, dtype=np.float32)

    def compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two normalized vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
