"""Vector Knowledge Store with dense search and metadata filtering."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from rag.embeddings.encoder import MedicalEmbeddingEngine
from rag.schemas import DocumentChunk

logger = logging.getLogger("medicine_ai.rag.vector_store")


class MedicalVectorStore:
    """In-memory and Qdrant-compatible vector index for clinical document chunks."""

    def __init__(self, embedding_engine: Optional[MedicalEmbeddingEngine] = None):
        self.embedding_engine = embedding_engine or MedicalEmbeddingEngine(embedding_dim=128)
        self.chunks: List[DocumentChunk] = []
        self.vectors: Optional[np.ndarray] = None
        self.chunk_id_map: Dict[str, DocumentChunk] = {}

    def add_chunk(self, chunk: DocumentChunk) -> None:
        """Embed and index a single document chunk."""
        self.add_chunks([chunk])

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Embed and index a list of document chunks."""
        if not chunks:
            return

        texts = [f"{c.title} - {c.section_name}: {c.text}" for c in chunks]
        new_vectors = self.embedding_engine.encode(texts)

        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

        self.chunks.extend(chunks)
        for c in chunks:
            self.chunk_id_map[c.chunk_id] = c

        logger.info(f"Indexed {len(chunks)} chunks in MedicalVectorStore (Total: {len(self.chunks)}).")

    def load_from_json(self, chunks_json_path: Path) -> None:
        """Load and index chunks directly from processed JSON file."""
        if not chunks_json_path.exists():
            logger.warning(f"Chunks file {chunks_json_path} does not exist.")
            return

        with open(chunks_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        chunks = [DocumentChunk(**item) for item in raw_data]
        self.add_chunks(chunks)

    def search_dense(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Tuple[float, DocumentChunk]]:
        """Perform dense vector search with metadata filtering."""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        q_vec = self.embedding_engine.encode(query)[0]
        # Dot product with normalized vectors equals cosine similarity
        similarities = np.dot(self.vectors, q_vec)

        # Apply filters
        candidate_scores = []
        for idx, score in enumerate(similarities):
            chunk = self.chunks[idx]
            if filters:
                match = True
                for k, v in filters.items():
                    if k == "rxcui" and chunk.rxcui != v:
                        match = False
                        break
                    elif k == "ingredient" and chunk.ingredient.lower() != str(v).lower():
                        match = False
                        break
                    elif k == "section_category" and chunk.section_category != v:
                        match = False
                        break
                if not match:
                    continue

            # Scale to [0.0, 1.0]
            normalized_score = float(max(0.0, min(1.0, (score + 1.0) / 2.0)))
            candidate_scores.append((normalized_score, chunk))

        candidate_scores.sort(key=lambda x: x[0], reverse=True)
        return candidate_scores[:top_k]
