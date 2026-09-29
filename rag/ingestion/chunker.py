"""Clinical Section-Aware Document Chunker for Phase 10A Knowledge Base.

Preserves clinical section boundaries (Indications, Dosage, Contraindications,
Adverse Reactions, Drug Interactions, Pregnancy, Renal, Geriatric, etc.)
without arbitrary character slicing.
"""
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple
from rag.schemas import DocumentChunk, RawMedicalDocument

# Canonical Clinical Section Taxonomy
SECTION_TAXONOMY: Dict[str, Dict[str, str]] = {
    "indications": {"name": "Indications & Clinical Usage", "category": "INDICATIONS"},
    "dosage": {"name": "Dosage & Administration Guidelines", "category": "DOSAGE_AND_ADMINISTRATION"},
    "contraindications": {"name": "Contraindications & Black Box Warnings", "category": "CONTRAINDICATIONS"},
    "warnings": {"name": "Warnings & Precautions", "category": "WARNINGS"},
    "precautions": {"name": "Precautions & Monitoring Measures", "category": "PRECAUTIONS"},
    "adverse_reactions": {"name": "Adverse Reactions & Side Effects", "category": "ADVERSE_REACTIONS"},
    "drug_interactions": {"name": "Drug-Drug & Food Interactions", "category": "DRUG_INTERACTIONS"},
    "pregnancy": {"name": "Pregnancy, Teratogenicity & Labor", "category": "PREGNANCY"},
    "lactation": {"name": "Lactation & Nursing Mothers", "category": "LACTATION"},
    "pediatric": {"name": "Pediatric Usage & Dosage Adjustments", "category": "PEDIATRIC_USE"},
    "geriatric": {"name": "Geriatric Usage & Age Considerations", "category": "GERIATRIC_USE"},
    "renal_impairment": {"name": "Renal Impairment & Dose Adjustments", "category": "RENAL_IMPAIRMENT"},
    "hepatic_impairment": {"name": "Hepatic Impairment & Monitoring", "category": "HEPATIC_IMPAIRMENT"},
    "patient_counseling": {"name": "Patient Counseling & Instructions", "category": "PATIENT_COUNSELING"},
    "overdosage": {"name": "Overdosage Symptoms & Management", "category": "OVERDOSAGE"},
    "pharmacology": {"name": "Clinical Pharmacology & Mechanism of Action", "category": "CLINICAL_PHARMACOLOGY"},
    "other": {"name": "Other Clinical Information", "category": "OTHER"}
}


def compute_content_hash(text: str) -> str:
    """Deterministic SHA-256 hash of chunk content."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def normalize_section_key(key: str) -> Tuple[str, str]:
    """Normalize raw section key to canonical title and category."""
    clean_k = key.lower().strip().replace(" ", "_").replace("-", "_")
    
    for canon_k, info in SECTION_TAXONOMY.items():
        if canon_k in clean_k or clean_k in canon_k:
            return info["name"], info["category"]
            
    # Default fallback
    title = key.replace("_", " ").title()
    return title, "OTHER"


class MedicalSectionChunker:
    """Chunks structured drug monographs preserving clinical section integrity."""

    def __init__(self, max_chunk_chars: int = 1500, chunk_overlap_chars: int = 150):
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars

    def chunk_document(self, doc: RawMedicalDocument) -> List[DocumentChunk]:
        """Split a raw medical monograph into section-aware, provenance-linked chunks."""
        chunks: List[DocumentChunk] = []
        retrieved_ts = getattr(doc, "retrieved_at", "2026-09-29T00:00:00Z") or "2026-09-29T00:00:00Z"

        for section_key, section_text in doc.sections.items():
            cleaned_text = re.sub(r"\s+", " ", section_text).strip()
            if not cleaned_text:
                continue

            section_title, section_category = normalize_section_key(section_key)

            # If section text fits in a single chunk, keep it intact
            if len(cleaned_text) <= self.max_chunk_chars:
                chunk_id = f"{doc.source_id}_{section_key.lower()}_0"
                c_hash = compute_content_hash(cleaned_text)
                
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=doc.source_id,
                    title=doc.title,
                    source_name=doc.source_name,
                    source_url=doc.source_url,
                    section_name=section_title,
                    section_category=section_key.lower(),
                    rxcui=doc.rxcui,
                    ingredient=doc.ingredient,
                    drug_class=doc.drug_class,
                    brand_aliases=doc.brand_aliases,
                    text=cleaned_text,
                    content_hash=c_hash,
                    metadata={
                        "chunk_id": chunk_id,
                        "medicine_name": doc.title,
                        "rxcui": doc.rxcui,
                        "ingredient": doc.ingredient,
                        "source_authority": doc.source_name,
                        "source_document": doc.source_id,
                        "source_version": doc.version_date,
                        "section": section_category,
                        "original_section": section_key,
                        "text": cleaned_text,
                        "sha256": c_hash,
                        "provenance_id": f"{doc.source_id}_{c_hash[:8]}",
                        "retrieved_at": retrieved_ts,
                        "atc_code": doc.atc_code,
                        "drug_class": doc.drug_class,
                        "brand_aliases": doc.brand_aliases,
                        "character_length": len(cleaned_text)
                    }
                ))
            else:
                # Split large section cleanly on sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
                current_chunk_sentences = []
                current_length = 0
                sub_idx = 0

                for sent in sentences:
                    sent_len = len(sent)
                    if current_length + sent_len > self.max_chunk_chars and current_chunk_sentences:
                        chunk_body = " ".join(current_chunk_sentences).strip()
                        chunk_id = f"{doc.source_id}_{section_key.lower()}_{sub_idx}"
                        c_hash = compute_content_hash(chunk_body)

                        chunks.append(DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=doc.source_id,
                            title=doc.title,
                            source_name=doc.source_name,
                            source_url=doc.source_url,
                            section_name=section_title,
                            section_category=section_key.lower(),
                            rxcui=doc.rxcui,
                            ingredient=doc.ingredient,
                            drug_class=doc.drug_class,
                            brand_aliases=doc.brand_aliases,
                            text=chunk_body,
                            content_hash=c_hash,
                            metadata={
                                "chunk_id": chunk_id,
                                "medicine_name": doc.title,
                                "rxcui": doc.rxcui,
                                "ingredient": doc.ingredient,
                                "source_authority": doc.source_name,
                                "source_document": doc.source_id,
                                "source_version": doc.version_date,
                                "section": section_category,
                                "original_section": section_key,
                                "text": chunk_body,
                                "sha256": c_hash,
                                "provenance_id": f"{doc.source_id}_{c_hash[:8]}",
                                "retrieved_at": retrieved_ts,
                                "atc_code": doc.atc_code,
                                "drug_class": doc.drug_class,
                                "brand_aliases": doc.brand_aliases,
                                "character_length": len(chunk_body),
                                "sub_chunk_index": sub_idx
                            }
                        ))
                        sub_idx += 1
                        current_chunk_sentences = [current_chunk_sentences[-1]] if self.chunk_overlap_chars > 0 else []
                        current_length = sum(len(s) for s in current_chunk_sentences)

                    current_chunk_sentences.append(sent)
                    current_length += sent_len + 1

                if current_chunk_sentences:
                    chunk_body = " ".join(current_chunk_sentences).strip()
                    chunk_id = f"{doc.source_id}_{section_key.lower()}_{sub_idx}"
                    c_hash = compute_content_hash(chunk_body)

                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=doc.source_id,
                        title=doc.title,
                        source_name=doc.source_name,
                        source_url=doc.source_url,
                        section_name=section_title,
                        section_category=section_key.lower(),
                        rxcui=doc.rxcui,
                        ingredient=doc.ingredient,
                        drug_class=doc.drug_class,
                        brand_aliases=doc.brand_aliases,
                        text=chunk_body,
                        content_hash=c_hash,
                        metadata={
                            "chunk_id": chunk_id,
                            "medicine_name": doc.title,
                            "rxcui": doc.rxcui,
                            "ingredient": doc.ingredient,
                            "source_authority": doc.source_name,
                            "source_document": doc.source_id,
                            "source_version": doc.version_date,
                            "section": section_category,
                            "original_section": section_key,
                            "text": chunk_body,
                            "sha256": c_hash,
                            "provenance_id": f"{doc.source_id}_{c_hash[:8]}",
                            "retrieved_at": retrieved_ts,
                            "atc_code": doc.atc_code,
                            "drug_class": doc.drug_class,
                            "brand_aliases": doc.brand_aliases,
                            "character_length": len(chunk_body),
                            "sub_chunk_index": sub_idx
                        }
                    ))

        return chunks
