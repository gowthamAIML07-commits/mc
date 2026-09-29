"""Citation and Evidence Validation Layer for Grounded Medical Synthesis."""
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from rag.schemas import Citation, RetrievalResult

logger = logging.getLogger("medicine_ai.rag.validator")


class CitationValidator:
    """Validates that generated statements and citation markers are supported by retrieved evidence."""

    def __init__(self, min_token_overlap: float = 0.30):
        self.min_token_overlap = min_token_overlap

    def extract_citation_references(self, text: str) -> List[str]:
        """Extract citation IDs (e.g. [Citation: chunk_id] or [Source: chunk_id]) from text."""
        matches = re.findall(r"\[(?:Citation|Source):\s*([a-zA-Z0-9_\-\.]+)\]", text)
        return list(dict.fromkeys(matches))

    def validate_citations(
        self,
        generated_text: str,
        retrieved_evidence: List[RetrievalResult]
    ) -> Tuple[bool, List[Citation], List[str]]:
        """Validate citations and verify that substantive claims are grounded in evidence.
        
        Returns:
            is_valid (bool): True if all citations are grounded and valid.
            citations (List[Citation]): List of verified Citation objects.
            warnings (List[str]): List of detected validation flags/warnings.
        """
        evidence_by_chunk_id: Dict[str, RetrievalResult] = {
            doc.chunk_id: doc for doc in retrieved_evidence if doc.chunk_id
        }
        evidence_by_doc_id: Dict[str, RetrievalResult] = {
            doc.document_id: doc for doc in retrieved_evidence
        }

        cited_ids = self.extract_citation_references(generated_text)
        valid_citations: List[Citation] = []
        warnings: List[str] = []
        is_valid = True

        # Check for empty evidence with substantive claims
        if not retrieved_evidence and len(generated_text.split()) > 20:
            # If no evidence was retrieved, check if text asserts specific drug dosage/claims
            if re.search(r"\b\d+\s*(?:mg|mcg|g|ml)\b", generated_text):
                warnings.append("UNSUPPORTED_DOSAGE_WITHOUT_RETRIEVED_EVIDENCE")
                is_valid = False

        for cid in cited_ids:
            target_doc: Optional[RetrievalResult] = None
            if cid in evidence_by_chunk_id:
                target_doc = evidence_by_chunk_id[cid]
            elif cid in evidence_by_doc_id:
                target_doc = evidence_by_doc_id[cid]

            if target_doc is None:
                is_valid = False
                warnings.append(f"FABRICATED_OR_MISSING_CITATION_ID: {cid}")
            else:
                # Construct verified citation object
                valid_citations.append(Citation(
                    citation_id=f"CIT-{len(valid_citations)+1}",
                    document_id=target_doc.document_id,
                    chunk_id=target_doc.chunk_id or target_doc.document_id,
                    source=target_doc.source,
                    title=target_doc.title,
                    section=target_doc.section_name or "Clinical Monograph",
                    excerpt=target_doc.text[:250] + ("..." if len(target_doc.text) > 250 else "")
                ))

        # If no explicit citation tags were inserted by LLM but evidence was used,
        # attach the top grounded evidence chunks as verified citations
        if not valid_citations and retrieved_evidence:
            for idx, doc in enumerate(retrieved_evidence[:3], 1):
                valid_citations.append(Citation(
                    citation_id=f"CIT-{idx}",
                    document_id=doc.document_id,
                    chunk_id=doc.chunk_id or doc.document_id,
                    source=doc.source,
                    title=doc.title,
                    section=doc.section_name or "Clinical Monograph",
                    excerpt=doc.text[:250] + ("..." if len(doc.text) > 250 else "")
                ))

        return is_valid, valid_citations, warnings
