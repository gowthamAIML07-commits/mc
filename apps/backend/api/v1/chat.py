"""Chat API Endpoint providing Evidence-Grounded Medical AI Conversations."""
import logging
import uuid
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status

from ml.intent.classifier import MedicalIntentClassifier
from ml.ner.extractor import MedicalEntityExtractor
from ml.safety.guardrails import MedicalSafetyGuardrails
from rag.interactions.engine import DrugInteractionEngine
from rag.llm.generator import MedicalLLM
from rag.reranking.reranker import ClinicalCrossEncoderReranker
from rag.retrieval.hybrid import HybridMedicalRetriever
from rag.schemas import ChatRequest, ChatResponse, DrugInteractionResult, MedicalEntity, RetrievalResult
from rag.validation.citation_validator import CitationValidator

logger = logging.getLogger("medicine_ai.chat")

router = APIRouter(prefix="/chat", tags=["Clinical Chatbot Engine"])

# Initialize clinical pipeline singletons
entity_extractor = MedicalEntityExtractor()
intent_classifier = MedicalIntentClassifier()
safety_guardrails = MedicalSafetyGuardrails()
retriever = HybridMedicalRetriever()
reranker = ClinicalCrossEncoderReranker()
interaction_engine = DrugInteractionEngine()
llm_generator = MedicalLLM()
citation_validator = CitationValidator()


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def handle_chat_message(request: ChatRequest) -> ChatResponse:
    """Process a medical inquiry through the complete grounded RAG clinical pipeline."""
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Chat message cannot be empty or whitespace only."
        )

    conversation_id = request.conversation_id or str(uuid.uuid4())
    user_text = request.message.strip()
    warnings: List[str] = []

    # 1. Medical Entity Extraction
    entities = entity_extractor.extract_entities(user_text)

    # 2. Clinical Intent Classification
    intent = intent_classifier.classify_intent(user_text)

    # 3. Safety Guardrails & Emergency Classification
    safety = safety_guardrails.assess_safety(user_text, intent=intent, entities=entities)

    # 4. Emergency & Critical Risk Short-Circuit
    if safety.is_emergency or not safety.is_safe_to_generate:
        logger.warning(f"Emergency / Safety block triggered for conversation {conversation_id}: {safety.safety_flags}")
        return ChatResponse(
            response=safety.emergency_guidance or "Emergency medical assistance required. Please contact emergency services immediately.",
            conversation_id=conversation_id,
            intent="EMERGENCY",
            safety_level=safety.safety_level,
            citations=[],
            evidence=[],
            entities=entities,
            interactions=[],
            requires_professional_review=True,
            warnings=safety.safety_flags
        )

    # 5. Extract and Validate Prescription Context
    prescription_drugs: List[str] = []
    has_unverified_prescriptions = False
    if request.prescription_context:
        candidates = request.prescription_context.get("candidates", [])
        for c in candidates:
            status_val = c.get("verification_status")
            if status_val == "verified":
                name = c.get("normalized_name") or c.get("raw_text")
                if name:
                    prescription_drugs.append(name)
            elif status_val in ["unverified", "review_required"]:
                has_unverified_prescriptions = True
                warnings.append(f"Unverified prescription candidate '{c.get('raw_text')}' requires pharmacist inspection.")

    # 6. Structured Drug-Drug Interaction Lookup
    # Gather detected drug entities + verified prescription drugs
    detected_drugs = [e.canonical_name or e.text for e in entities if e.label in ["DRUG", "BRAND", "ACTIVE_INGREDIENT"]]
    all_drugs = list(dict.fromkeys(detected_drugs + prescription_drugs))

    interactions: List[DrugInteractionResult] = []
    if intent == "DRUG_INTERACTION" or len(all_drugs) >= 2:
        if len(all_drugs) >= 2:
            interactions = interaction_engine.check_all_interactions(all_drugs)
        elif len(all_drugs) == 1:
            # Single drug asking for interactions: check against known general interaction rules
            pass

    # 7. Knowledge Retrieval & Reranking
    retrieved_evidence: List[RetrievalResult] = []
    reranked_evidence: List[RetrievalResult] = []

    if safety.safety_level != "OUT_OF_SCOPE":
        # Execute hybrid retrieval
        retrieved_evidence = await retriever.search(user_text, top_k=6)
        # Execute cross-encoder reranking
        reranked_evidence = await reranker.rerank(user_text, retrieved_evidence, top_n=3)

    # 8. Grounded LLM Response Synthesis
    generated_text = llm_generator.generate_response(
        query=user_text,
        intent=intent,
        safety=safety,
        entities=entities,
        evidence=reranked_evidence,
        interactions=interactions,
        prescription_context=request.prescription_context
    )

    # 9. Citation & Evidence Validation
    is_valid, citations, val_warnings = citation_validator.validate_citations(
        generated_text, reranked_evidence
    )
    warnings.extend(val_warnings)

    requires_review = (
        safety.safety_level in ["HIGH", "EMERGENCY"] or
        has_unverified_prescriptions or
        any(i.severity in ["MAJOR", "CONTRAINDICATED"] for i in interactions) or
        intent in ["CONTRAINDICATION", "DOSAGE_INFORMATION", "PREGNANCY"]
    )

    return ChatResponse(
        response=generated_text,
        conversation_id=conversation_id,
        intent=intent,
        safety_level=safety.safety_level,
        citations=citations,
        evidence=reranked_evidence,
        entities=entities,
        interactions=interactions,
        requires_professional_review=requires_review,
        warnings=warnings
    )
