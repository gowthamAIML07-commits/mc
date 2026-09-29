"""Hybrid Medical Knowledge Retriever combining Lexical, Structured Entity, and Dense Vector Search."""
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import Levenshtein

from apps.backend.core.config import settings
from rag.indexing.vector_store import MedicalVectorStore
from rag.interfaces import MedicalKnowledgeRetriever
from rag.schemas import DocumentChunk, RetrievalResult

logger = logging.getLogger("medicine_ai.rag.retrieval")


class HybridMedicalRetriever(MedicalKnowledgeRetriever):
    """Production hybrid retriever fusing lexical keyword match, entity routing, and dense embeddings."""

    def __init__(self, vector_store: Optional[MedicalVectorStore] = None, chunks_path: Optional[Path] = None):
        self.vector_store = vector_store or MedicalVectorStore()
        if chunks_path is None:
            norm_kb = Path(settings.NORMALIZATION_KB_PATH)
            # Check in data/processed/knowledge_base/clinical_chunks.json
            possible_path = norm_kb.parent.parent / "knowledge_base" / "clinical_chunks.json"
            if not possible_path.exists():
                possible_path = norm_kb.parent / "knowledge_base" / "clinical_chunks.json"
            chunks_path = possible_path

        if chunks_path.exists() and len(self.vector_store.chunks) == 0:
            self.vector_store.load_from_json(chunks_path)

    def _lexical_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 15
    ) -> List[Tuple[float, DocumentChunk]]:
        """Perform lexical, clinical entity, and section keyword matching."""
        q_lower = query.lower()
        q_tokens = set(re.findall(r"\b\w+\b", q_lower))
        scored = []

        for chunk in self.vector_store.chunks:
            if filters:
                match = True
                for k, v in filters.items():
                    if k == "rxcui" and chunk.rxcui != v:
                        match = False
                        break
                    elif k == "ingredient" and chunk.ingredient.lower() != str(v).lower():
                        match = False
                        break
                if not match:
                    continue

            score = 0.0
            chunk_text_lower = chunk.text.lower()
            chunk_tokens = set(re.findall(r"\b\w+\b", chunk_text_lower))

            # 1. Exact drug ingredient / brand match
            if chunk.ingredient.lower() in q_lower:
                score += 0.45
            for alias in chunk.brand_aliases:
                if alias.lower() in q_lower:
                    score += 0.40
                    break

            # 2. Section relevance keyword match
            sec_name_lower = chunk.section_name.lower()
            if any(term in q_lower for term in ["dose", "dosage", "take", "how many", "schedule"]) and "dosage" in sec_name_lower:
                score += 0.35
            elif any(term in q_lower for term in ["side effect", "adverse", "reaction", "harm"]) and "adverse" in sec_name_lower:
                score += 0.35
            elif any(term in q_lower for term in ["contraindication", "avoid", "danger", "who should not"]) and "contraindication" in sec_name_lower:
                score += 0.35
            elif any(term in q_lower for term in ["interact", "combination", "together", "with"]) and "interaction" in sec_name_lower:
                score += 0.35
            elif any(term in q_lower for term in ["treat", "indication", "prescribed for", "use"]) and "indication" in sec_name_lower:
                score += 0.30
            elif any(term in q_lower for term in ["pregnant", "pregnancy", "breastfeed", "lactat"]) and "pregnancy" in sec_name_lower:
                score += 0.40

            # 3. Jaccard token overlap
            overlap = len(q_tokens.intersection(chunk_tokens))
            jaccard = overlap / max(len(q_tokens), 1)
            score += 0.20 * jaccard

            if score > 0.15:
                scored.append((min(round(score, 4), 1.0), chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Execute hybrid search combining dense vector similarity and structured lexical scores."""
        # 1. Dense Semantic Vector Search
        dense_results = self.vector_store.search_dense(query, filters=filters, top_k=top_k * 2)

        # 2. Lexical & Entity Keyword Search
        lexical_results = self._lexical_search(query, filters=filters, top_k=top_k * 2)

        # 3. Reciprocal Rank Fusion (RRF) & Score Combination
        fused_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, DocumentChunk] = {}
        rrf_k = 60

        for rank, (score, chunk) in enumerate(lexical_results, 1):
            chunk_lookup[chunk.chunk_id] = chunk
            # High-confidence lexical matches receive strong priority
            lex_weight = 3.0 if score >= 0.4 else 1.5
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + lex_weight * (1.0 / (rrf_k + rank)) + score

        for rank, (score, chunk) in enumerate(dense_results, 1):
            chunk_lookup[chunk.chunk_id] = chunk
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + 1.0 * (1.0 / (rrf_k + rank))

        # Sort descending by fused score
        sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        results: List[RetrievalResult] = []
        for chunk_id, fused_score in sorted_chunks[:top_k]:
            c = chunk_lookup[chunk_id]
            norm_score = min(round(fused_score / 1.5, 4), 1.0)
            
            results.append(RetrievalResult(
                document_id=c.document_id,
                chunk_id=c.chunk_id,
                title=c.title,
                source=c.source_name,
                source_url=c.source_url,
                section_name=c.section_name,
                text=c.text,
                score=norm_score,
                retrieval_method="hybrid_rrf",
                metadata={
                    "rxcui": c.rxcui,
                    "ingredient": c.ingredient,
                    "drug_class": c.drug_class,
                    "brand_aliases": c.brand_aliases,
                    "section_category": c.section_category,
                    "content_hash": c.content_hash
                }
            ))

        return results
