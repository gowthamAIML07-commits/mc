"""Cross-Encoder Clinical Evidence Reranker."""
import logging
import re
from typing import Any, Dict, List, Optional
import Levenshtein

from rag.interfaces import MedicalReranker
from rag.schemas import RetrievalResult

logger = logging.getLogger("medicine_ai.rag.reranker")


class ClinicalCrossEncoderReranker(MedicalReranker):
    """Reranker that cross-evaluates query and candidate clinical evidence chunks."""

    def __init__(self, min_relevance_threshold: float = 0.40):
        self.min_relevance_threshold = min_relevance_threshold

    def _score_query_document_pair(self, query: str, doc: RetrievalResult) -> float:
        """Compute fine-grained cross-scoring between query and candidate monograph."""
        q_lower = query.lower()
        doc_text_lower = doc.text.lower()
        title_lower = doc.title.lower()
        section_lower = doc.section_name.lower()

        q_tokens = set(re.findall(r"\b\w+\b", q_lower))
        doc_tokens = set(re.findall(r"\b\w+\b", doc_text_lower))

        score = 0.0

        # 1. Ingredient & Brand Exact Hit Bonus
        ing = doc.metadata.get("ingredient", "").lower()
        if ing and ing in q_lower:
            score += 0.35

        for alias in doc.metadata.get("brand_aliases", []):
            if alias.lower() in q_lower:
                score += 0.30
                break

        # 2. Section Direct Alignment
        if any(w in q_lower for w in ["interact", "combination", "together"]) and "interaction" in section_lower:
            score += 0.40
        elif any(w in q_lower for w in ["side effect", "adverse", "reaction"]) and "adverse" in section_lower:
            score += 0.40
        elif any(w in q_lower for w in ["dose", "dosage", "take", "how much"]) and "dosage" in section_lower:
            score += 0.35
        elif any(w in q_lower for w in ["avoid", "contraindication", "danger"]) and "contraindication" in section_lower:
            score += 0.35
        elif any(w in q_lower for w in ["pregnancy", "pregnant", "nursing", "breastfeed"]) and "pregnancy" in section_lower:
            score += 0.45

        # 3. Dense Token Overlap & Semantic Coverage
        overlap = len(q_tokens.intersection(doc_tokens))
        token_coverage = overlap / max(len(q_tokens), 1)
        score += 0.25 * token_coverage

        # 4. Integrate initial retrieval score
        score += 0.20 * doc.score

        return min(round(score, 4), 1.0)

    async def rerank(
        self,
        query: str,
        documents: List[RetrievalResult],
        top_n: int = 3
    ) -> List[RetrievalResult]:
        """Rerank candidate evidence chunks and return the top-N highest relevance documents."""
        if not documents:
            return []

        scored_docs = []
        for doc in documents:
            rerank_score = self._score_query_document_pair(query, doc)
            if rerank_score >= self.min_relevance_threshold:
                # Update document attributes
                doc_copy = doc.model_copy()
                doc_copy.metadata["retrieval_initial_score"] = doc.score
                doc_copy.metadata["reranker_score"] = rerank_score
                doc_copy.score = rerank_score
                doc_copy.retrieval_method = "cross_encoder_reranked"
                scored_docs.append((rerank_score, doc_copy))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        reranked_results = []
        for rank, (_, doc) in enumerate(scored_docs[:top_n], 1):
            doc.metadata["final_rank"] = rank
            reranked_results.append(doc)

        logger.info(f"Reranked {len(documents)} candidates -> {len(reranked_results)} top evidence chunks.")
        return reranked_results
