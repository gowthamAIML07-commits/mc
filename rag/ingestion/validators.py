"""Clinical Knowledge Ingestion Quality and Security Validators."""
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import ValidationError

from rag.schemas import DocumentChunk, RawMedicalDocument


class DocumentQualityStatus:
    VALID = "VALID"
    WARNING = "WARNING"
    REJECTED = "REJECTED"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


class ClinicalDocumentValidator:
    """Validates raw medical documents and chunks for integrity, security, and clinical completeness."""

    MIN_DOCUMENT_CHARS = 100
    MIN_SECTION_CHARS = 15
    REQUIRED_SECTIONS = ["indications", "dosage"]
    
    DISALLOWED_PATTERNS = [
        re.compile(r"<script.*?>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"onload\s*=", re.IGNORECASE),
        re.compile(r"onerror\s*=", re.IGNORECASE),
        re.compile(r"\x00"),  # Null byte injection
    ]

    @classmethod
    def validate_raw_document(cls, doc: RawMedicalDocument) -> Tuple[str, List[str]]:
        """Validate raw medical monograph before chunking.
        
        Returns:
            status: VALID, WARNING, REJECTED, or REQUIRES_REVIEW
            issues: list of validation messages / warnings
        """
        issues: List[str] = []

        # 1. Structural Checks
        if not doc.source_id or not doc.source_id.strip():
            return DocumentQualityStatus.REJECTED, ["Missing required source_id"]
        if not doc.title or not doc.title.strip():
            return DocumentQualityStatus.REJECTED, ["Missing required title"]
        if not doc.ingredient or not doc.ingredient.strip():
            return DocumentQualityStatus.REJECTED, ["Missing required active ingredient"]
        if not doc.source_name or not doc.source_name.strip():
            return DocumentQualityStatus.REJECTED, ["Missing required source_name authority"]

        # 2. Section Checks
        if not doc.sections or len(doc.sections) == 0:
            return DocumentQualityStatus.REJECTED, ["Document contains no clinical sections"]

        total_chars = sum(len(text) for text in doc.sections.values())
        if total_chars < cls.MIN_DOCUMENT_CHARS:
            return DocumentQualityStatus.REJECTED, [f"Suspiciously short document text ({total_chars} chars)"]

        # Check required sections
        missing_req = [sec for sec in cls.REQUIRED_SECTIONS if sec not in doc.sections or not doc.sections[sec].strip()]
        if missing_req:
            issues.append(f"Missing core clinical sections: {', '.join(missing_req)}")

        # 3. Security & Sanitization Checks
        for sec_name, sec_text in doc.sections.items():
            if not sec_text or len(sec_text.strip()) < cls.MIN_SECTION_CHARS:
                issues.append(f"Section '{sec_name}' is suspiciously short ({len(sec_text.strip()) if sec_text else 0} chars)")
            
            for pat in cls.DISALLOWED_PATTERNS:
                if pat.search(sec_text):
                    return DocumentQualityStatus.REJECTED, [f"Security violation: Disallowed script/binary pattern in section '{sec_name}'"]

        # 4. RxCUI and ATC format verification
        if not doc.rxcui or not str(doc.rxcui).isdigit():
            issues.append(f"RxCUI missing or non-standard format: '{doc.rxcui}'")

        # Determine overall status
        if any("Security violation" in i or "Missing required" in i for i in issues):
            return DocumentQualityStatus.REJECTED, issues
        elif issues:
            if any("Missing core clinical sections" in i for i in issues):
                return DocumentQualityStatus.REQUIRES_REVIEW, issues
            return DocumentQualityStatus.WARNING, issues
        
        return DocumentQualityStatus.VALID, []

    @classmethod
    def validate_chunk(cls, chunk: DocumentChunk) -> Tuple[str, List[str]]:
        """Validate an individual clinical document chunk."""
        issues: List[str] = []

        if not chunk.chunk_id or not chunk.chunk_id.strip():
            return DocumentQualityStatus.REJECTED, ["Missing chunk_id"]
        if not chunk.text or len(chunk.text.strip()) < 10:
            return DocumentQualityStatus.REJECTED, ["Chunk text is empty or too short"]
        if not chunk.content_hash or len(chunk.content_hash) != 64:
            return DocumentQualityStatus.REJECTED, ["Invalid or missing SHA-256 content_hash"]

        # Check hash match
        expected_hash = hashlib.sha256(chunk.text.strip().encode("utf-8")).hexdigest()
        if chunk.content_hash != expected_hash:
            return DocumentQualityStatus.REJECTED, [f"Content hash mismatch for chunk {chunk.chunk_id}"]

        for pat in cls.DISALLOWED_PATTERNS:
            if pat.search(chunk.text):
                return DocumentQualityStatus.REJECTED, ["Security violation: script/code pattern in chunk text"]

        return DocumentQualityStatus.VALID, []
