"""Abstract Interfaces and Service Boundaries for Medical Knowledge Retrieval & RAG."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from rag.schemas import EvidenceSet, RetrievalQuery, RetrievalResult


class MedicalKnowledgeRetriever(ABC):
    """Abstract interface for medical knowledge retrieval (Lexical, Dense, and Hybrid)."""

    @abstractmethod
    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Search medical monographs, active ingredients, and clinical guidelines."""
        pass


class MedicalReranker(ABC):
    """Abstract interface for cross-encoder reranking of retrieved clinical evidence."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[RetrievalResult],
        top_n: int = 3
    ) -> List[RetrievalResult]:
        """Score and reorder candidate evidence documents by clinical relevance."""
        pass
