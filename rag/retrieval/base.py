"""Concrete Baseline Retriever indexing the verified Phase 4 RxNorm/RxTerms Knowledge Base."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import Levenshtein

from apps.backend.core.config import settings
from rag.interfaces import MedicalKnowledgeRetriever
from rag.schemas import RetrievalResult

logger = logging.getLogger("medicine_ai.rag.retriever")


class InMemoryRxNormRetriever(MedicalKnowledgeRetriever):
    """Knowledge retriever backed by verified RxNorm concepts and RxTerms descriptors."""

    def __init__(self, kb_path: Optional[str] = None):
        self.kb_path = Path(kb_path or settings.NORMALIZATION_KB_PATH)
        self.documents: List[Dict[str, Any]] = []
        self._load_knowledge()

    def _load_knowledge(self):
        """Load verified RxNorm clinical concepts."""
        if not self.kb_path.exists():
            logger.warning(f"RxNorm knowledge base not found at {self.kb_path}. Initializing empty retriever.")
            return

        with open(self.kb_path, "r", encoding="utf-8") as f:
            concepts = json.load(f)

        for c in concepts:
            doc_text = (
                f"Medication: {c.get('name')}. Active Ingredient: {c.get('ingredient')}. "
                f"Strength: {c.get('strength')}. Form: {c.get('form')}. Route: {c.get('route')}. "
                f"Pharmacological Class: {c.get('drug_class')}. Known Brand Aliases: {', '.join(c.get('aliases', []))}."
            )
            self.documents.append({
                "document_id": f"RXCUI_{c.get('rxcui')}",
                "title": c.get("name"),
                "source": "US NLM RxNorm / RxTerms",
                "text": doc_text,
                "rxcui": c.get("rxcui"),
                "ingredient": c.get("ingredient"),
                "drug_class": c.get("drug_class"),
                "aliases": c.get("aliases", []),
                "raw_concept": c
            })
        logger.info(f"Loaded {len(self.documents)} clinical monographs into InMemoryRxNormRetriever.")

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Perform fuzzy and semantic keyword search over clinical concept monographs."""
        q_clean = query.lower().strip()
        scored_docs = []

        for doc in self.documents:
            # Apply filters if specified
            if filters:
                match_filter = True
                for k, v in filters.items():
                    if doc.get(k) != v:
                        match_filter = False
                        break
                if not match_filter:
                    continue

            # Compute lexical similarity against title, ingredient, and aliases
            sim_title = Levenshtein.ratio(q_clean, doc["title"].lower())
            sim_ing = Levenshtein.ratio(q_clean, doc["ingredient"].lower())
            sim_alias = max([Levenshtein.ratio(q_clean, a.lower()) for a in doc["aliases"]] or [0.0])

            score = max(sim_title, sim_ing, sim_alias)
            if q_clean in doc["title"].lower() or q_clean in doc["ingredient"].lower():
                score = max(score, 0.95)

            if score > 0.40:
                scored_docs.append((score, doc))

        # Sort descending by score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored_docs[:top_k]:
            results.append(RetrievalResult(
                document_id=doc["document_id"],
                title=doc["title"],
                source=doc["source"],
                text=doc["text"],
                score=round(score, 4),
                metadata={
                    "rxcui": doc["rxcui"],
                    "ingredient": doc["ingredient"],
                    "drug_class": doc["drug_class"]
                }
            ))

        return results
