"""Pydantic schemas for Medical Knowledge Retrieval, NER, Safety, and RAG Chatbot."""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# --- Ingestion & Provenance Schemas ---

class RawMedicalDocument(BaseModel):
    source_id: str = Field(..., description="Unique source document identifier (e.g. DailyMed Set ID or RxCUI)")
    source_name: str = Field(..., description="Name of knowledge source: DailyMed, RxNorm, FDA, WHO")
    source_type: str = Field("drug_label_monograph", description="Document type: monograph, guideline, terminology")
    source_url: Optional[str] = Field(None, description="Official authoritative reference URL")
    title: str = Field(..., description="Official drug brand and generic formulation title")
    version_date: Optional[str] = Field("2024-12", description="Publication or revision date")
    rxcui: Optional[str] = Field(None, description="RxNorm Concept Unique Identifier")
    ingredient: str = Field(..., description="Active pharmaceutical ingredient (INN)")
    drug_class: Optional[str] = Field(None, description="Pharmacological / therapeutic class")
    atc_code: Optional[str] = Field(None, description="Anatomical Therapeutic Chemical code")
    brand_aliases: List[str] = Field(default_factory=list, description="Recognized commercial brand names")
    sections: Dict[str, str] = Field(default_factory=dict, description="Structured clinical sections (Indications, Dosage, etc.)")


class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description="Deterministic unique chunk identifier: {source_id}_{section_key}_{idx}")
    document_id: str = Field(..., description="Parent document identifier")
    title: str = Field(..., description="Parent document title")
    source_name: str = Field(..., description="Knowledge authority source")
    source_url: Optional[str] = Field(None, description="Source reference URL")
    section_name: str = Field(..., description="Clinical section name: Indications, Dosage, Contraindications, etc.")
    section_category: str = Field(..., description="Standard section category tag")
    rxcui: Optional[str] = Field(None, description="RxNorm Concept ID")
    ingredient: str = Field(..., description="Active drug ingredient")
    drug_class: Optional[str] = Field(None, description="Pharmacological class")
    brand_aliases: List[str] = Field(default_factory=list, description="Associated trade brand names")
    text: str = Field(..., description="Cleaned, section-contained clinical text content")
    content_hash: str = Field(..., description="SHA-256 content hash of chunk text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional structured attributes")


# --- Retrieval & Reranking Schemas ---

class RetrievalResult(BaseModel):
    document_id: str = Field(..., description="Document identifier")
    chunk_id: Optional[str] = Field(None, description="Specific chunk identifier")
    title: str = Field(..., description="Drug monograph title")
    source: str = Field(..., description="Authority source")
    source_url: Optional[str] = Field(None, description="Authoritative citation URL")
    section_name: Optional[str] = Field("monograph", description="Clinical section name")
    text: str = Field(..., description="Clinical evidence text excerpt")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance similarity score")
    retrieval_method: str = Field("hybrid", description="Method used: lexical, dense_vector, hybrid, reranked")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Structured concept metadata")


class EvidenceSet(BaseModel):
    query: str
    results: List[RetrievalResult] = Field(default_factory=list)
    total_found: int = 0


class RetrievalQuery(BaseModel):
    query_text: str
    entities: List[str] = Field(default_factory=list)
    filters: Optional[Dict[str, Any]] = None
    top_k: int = Field(5, ge=1, le=50)


# --- NER & Intent Schemas ---

class MedicalEntity(BaseModel):
    text: str
    label: Literal[
        "DRUG", "BRAND", "ACTIVE_INGREDIENT", "DISEASE", "SYMPTOM",
        "DOSAGE", "FREQUENCY", "DURATION", "AGE", "PREGNANCY",
        "ALLERGY", "LAB_TEST", "MEDICAL_PROCEDURE"
    ]
    start_char: int
    end_char: int
    normalized_rxcui: Optional[str] = None
    canonical_name: Optional[str] = None
    confidence: float = 1.0


IntentType = Literal[
    "MEDICINE_INFORMATION",
    "MEDICINE_USE",
    "SIDE_EFFECT",
    "DOSAGE_INFORMATION",
    "CONTRAINDICATION",
    "DRUG_INTERACTION",
    "MISSED_DOSE",
    "PREGNANCY",
    "CHILD_MEDICATION",
    "ELDERLY_MEDICATION",
    "PRESCRIPTION",
    "MEDICINE_IDENTIFICATION",
    "SYMPTOM",
    "EMERGENCY",
    "GENERAL_HEALTH",
    "OUT_OF_SCOPE"
]


# --- Safety Schemas ---

SafetyLevel = Literal["LOW", "MODERATE", "HIGH", "EMERGENCY", "OUT_OF_SCOPE"]


class SafetyAssessment(BaseModel):
    safety_level: SafetyLevel
    is_emergency: bool = False
    is_safe_to_generate: bool = True
    safety_flags: List[str] = Field(default_factory=list)
    emergency_guidance: Optional[str] = None
    disclaimer: str = (
        "Medical Information Disclaimer: This assistant provides evidence-based drug information for educational purposes only. "
        "It is not a substitute for professional clinical advice, diagnosis, or treatment."
    )


# --- Drug Interaction Schemas ---

class DrugInteractionResult(BaseModel):
    drug_a: str
    drug_b: str
    drug_a_rxcui: Optional[str] = None
    drug_b_rxcui: Optional[str] = None
    interaction_found: bool = False
    severity: Optional[Literal["CONTRAINDICATED", "MAJOR", "MODERATE", "MINOR", "UNKNOWN"]] = None
    clinical_effect: Optional[str] = None
    mechanism: Optional[str] = None
    evidence_source: Optional[str] = None
    recommendation: Optional[str] = None


# --- Chat API Schemas ---

class Citation(BaseModel):
    citation_id: str
    document_id: str
    chunk_id: str
    source: str
    title: str
    section: str
    excerpt: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="User medical question or inquiry")
    conversation_id: Optional[str] = Field(None, description="Optional ongoing conversation ID")
    prescription_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional PrescriptionResult context from Phase 5 extraction"
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="Evidence-grounded medical explanation")
    conversation_id: str = Field(..., description="Conversation correlation ID")
    intent: IntentType = Field(..., description="Detected user intent")
    safety_level: SafetyLevel = Field(..., description="Safety assessment classification")
    citations: List[Citation] = Field(default_factory=list, description="Verified source citations")
    evidence: List[RetrievalResult] = Field(default_factory=list, description="Retrieved evidence chunks")
    entities: List[MedicalEntity] = Field(default_factory=list, description="Extracted clinical entities")
    interactions: List[DrugInteractionResult] = Field(default_factory=list, description="Verified drug interaction findings")
    requires_professional_review: bool = Field(False, description="Flag indicating user must consult healthcare provider")
    warnings: List[str] = Field(default_factory=list, description="Safety and verification warnings")
