"""Comprehensive Unit Tests for Phase 6 Medical Knowledge RAG & Clinical Chatbot Engine."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ml.embeddings.normalizer import MedicineNormalizer
from ml.intent.classifier import MedicalIntentClassifier
from ml.ner.extractor import MedicalEntityExtractor
from ml.safety.guardrails import MedicalSafetyGuardrails
from rag.embeddings.encoder import MedicalEmbeddingEngine
from rag.evaluation.evaluator import MedicalRAGEvaluator
from rag.indexing.vector_store import MedicalVectorStore
from rag.ingestion.adapter import DailyMedRxNormIngestionAdapter
from rag.ingestion.chunker import MedicalSectionChunker
from rag.interactions.engine import DrugInteractionEngine
from rag.llm.generator import MedicalLLM
from rag.reranking.reranker import ClinicalCrossEncoderReranker
from rag.retrieval.hybrid import HybridMedicalRetriever
from rag.schemas import (
    Citation,
    DocumentChunk,
    DrugInteractionResult,
    MedicalEntity,
    RawMedicalDocument,
    RetrievalResult,
    SafetyAssessment
)
from rag.validation.citation_validator import CitationValidator


# 1. Document Ingestion & Raw Validation
def test_document_ingestion_adapter():
    adapter = DailyMedRxNormIngestionAdapter()
    docs = adapter.build_authoritative_monographs()
    assert len(docs) >= 7
    for doc in docs:
        assert isinstance(doc, RawMedicalDocument)
        assert doc.source_id.startswith("DAILYMED_")
        assert "DailyMed" in doc.source_name
        assert len(doc.sections) >= 3


# 2. Metadata Preservation
def test_metadata_preservation():
    adapter = DailyMedRxNormIngestionAdapter()
    docs = adapter.build_authoritative_monographs()
    warfarin = next(d for d in docs if d.ingredient.lower() == "warfarin")
    assert warfarin.rxcui == "11289"
    assert warfarin.atc_code == "B01AA03"
    assert "Coumadin" in warfarin.brand_aliases
    assert "indications" in warfarin.sections


# 3. Document Hashing & 4. Duplicate Detection & 5. Chunking
def test_medical_section_chunking_and_hashing():
    chunker = MedicalSectionChunker(max_chunk_chars=400, chunk_overlap_chars=50)
    raw_doc = RawMedicalDocument(
        source_id="DAILYMED_TEST",
        source_name="DailyMed",
        title="Test Drug 500mg",
        ingredient="Testicillin",
        rxcui="99999",
        sections={
            "indications": "Testicillin is indicated for acute bacterial pharyngitis.",
            "dosage": "Administer 500 mg orally every 8 hours for 10 days.",
            "contraindications": "Hypersensitivity to beta-lactam antibiotics."
        }
    )
    chunks = chunker.chunk_document(raw_doc)
    assert len(chunks) == 3
    for c in chunks:
        assert isinstance(c, DocumentChunk)
        assert len(c.content_hash) == 64  # SHA-256
        assert c.ingredient == "Testicillin"
        assert c.document_id == "DAILYMED_TEST"

    # Duplicate detection
    dup_chunks = chunker.chunk_document(raw_doc)
    assert len(dup_chunks) == len(chunks)
    assert dup_chunks[0].content_hash == chunks[0].content_hash


# 6. Embedding Generation
def test_medical_embedding_generation():
    engine = MedicalEmbeddingEngine(embedding_dim=128)
    vec = engine.encode("Warfarin Sodium 5 mg Oral Tablet")[0]
    assert len(vec) == 128
    # Test L2 normalization
    norm = sum(x**2 for x in vec)**0.5
    assert pytest.approx(norm, rel=1e-3) == 1.0


# 7. Vector Indexing & 9. Dense Retrieval
def test_vector_indexing_and_dense_search(tmp_path):
    store = MedicalVectorStore()
    chunk = DocumentChunk(
        chunk_id="TEST_001",
        document_id="DOC_001",
        title="Metformin Hydrochloride Monograph",
        source_name="DailyMed",
        section_name="Indications",
        section_category="indications",
        rxcui="860975",
        ingredient="Metformin",
        brand_aliases=["Glucophage"],
        text="Metformin is indicated as an adjunct to diet and exercise to improve glycemic control in type 2 diabetes mellitus.",
        content_hash="a" * 64
    )
    store.add_chunk(chunk)
    assert len(store.chunks) == 1

    # Dense search
    results = store.search_dense("type 2 diabetes glycemic control", top_k=1)
    assert len(results) == 1
    score, res_chunk = results[0]
    assert res_chunk.ingredient == "Metformin"
    assert score > 0.0


# 8. Lexical Retrieval & 10. Hybrid Retrieval
@pytest.mark.asyncio
async def test_hybrid_retrieval_fusion():
    retriever = HybridMedicalRetriever()
    results = await retriever.search("warfarin bleeding risk interactions", top_k=3)
    assert len(results) >= 1
    top_hit = results[0]
    assert "Warfarin" in top_hit.title or "warfarin" in top_hit.text.lower()
    assert top_hit.retrieval_method == "hybrid_rrf"
    assert top_hit.score > 0.0


# 11. Cross-Encoder Reranking
@pytest.mark.asyncio
async def test_cross_encoder_reranker():
    reranker = ClinicalCrossEncoderReranker()
    candidate = RetrievalResult(
        document_id="DAILYMED_WARFARIN",
        chunk_id="DAILYMED_WARFARIN_drug_interactions_0",
        title="Warfarin Sodium Tablet",
        source="DailyMed",
        section_name="Drug Interactions",
        text="Co-administration of warfarin with aspirin or NSAIDs significantly increases gastrointestinal bleeding.",
        score=0.75,
        metadata={"ingredient": "Warfarin", "brand_aliases": ["Coumadin"]}
    )
    reranked = await reranker.rerank("warfarin aspirin interaction", [candidate], top_n=1)
    assert len(reranked) == 1
    assert reranked[0].retrieval_method == "cross_encoder_reranked"
    assert reranked[0].metadata["reranker_score"] > 0.50
    assert reranked[0].metadata["final_rank"] == 1


# 12. Entity Extraction Integration & 13. RxNorm Normalization
def test_entity_extraction_and_rxnorm_normalization():
    extractor = MedicalEntityExtractor()
    text = "Patient taking Metformin 500mg twice daily for type 2 diabetes and having nausea."
    entities = extractor.extract_entities(text)

    labels = {e.label for e in entities}
    assert "DRUG" in labels
    assert "DOSAGE" in labels
    assert "FREQUENCY" in labels
    assert "DISEASE" in labels
    assert "SYMPTOM" in labels

    drug_entity = next(e for e in entities if e.label == "DRUG")
    assert drug_entity.text.lower() == "metformin"
    assert drug_entity.normalized_rxcui == "860975"
    assert drug_entity.canonical_name == "Metformin hydrochloride 500 MG Oral Tablet"


# 14. Intent Classification
def test_intent_classification():
    classifier = MedicalIntentClassifier()
    assert classifier.classify_intent("Can I take aspirin with warfarin?") == "DRUG_INTERACTION"
    assert classifier.classify_intent("What are the side effects of atorvastatin?") == "SIDE_EFFECT"
    assert classifier.classify_intent("What is the dose for amoxicillin in kids?") == "CHILD_MEDICATION"
    assert classifier.classify_intent("I have severe chest pain and cannot breathe") == "EMERGENCY"
    assert classifier.classify_intent("Is paracetamol safe during pregnancy?") == "PREGNANCY"
    assert classifier.classify_intent("Who should not take pantoprazole?") == "CONTRAINDICATION"


# 15. Safety Classification & 16. Emergency Routing
def test_safety_guardrails_emergency_routing():
    guardrails = MedicalSafetyGuardrails()
    
    # Emergency trigger
    emergency_eval = guardrails.assess_safety("I took 30 sleeping pills and drank poison")
    assert emergency_eval.safety_level == "EMERGENCY"
    assert emergency_eval.is_emergency is True
    assert emergency_eval.is_safe_to_generate is False
    assert "CRITICAL MEDICAL ALERT" in emergency_eval.emergency_guidance

    # Prescribing attempt trigger
    prescribe_eval = guardrails.assess_safety("Prescribe me 500mg Amoxicillin right now")
    assert prescribe_eval.safety_level == "HIGH"
    assert "PRESCRIBING_OR_DIAGNOSIS_ATTEMPT" in prescribe_eval.safety_flags

    # Safe query
    safe_eval = guardrails.assess_safety("What is the mechanism of action of metformin?")
    assert safe_eval.safety_level == "LOW"
    assert safe_eval.is_safe_to_generate is True


# 17. Unsupported Medical Claim Detection & 18. Citation Validation
def test_citation_validation():
    validator = CitationValidator()
    evidence = [
        RetrievalResult(
            document_id="DOC_01",
            chunk_id="CHUNK_01",
            title="Metformin Monograph",
            source="DailyMed",
            section_name="Dosage",
            text="The recommended starting dose of Metformin is 500 mg orally twice daily with meals.",
            score=0.90
        )
    ]

    # Valid citation
    valid_text = "Metformin is taken with meals [Source: CHUNK_01]."
    is_valid, citations, warnings = validator.validate_citations(valid_text, evidence)
    assert is_valid is True
    assert len(citations) == 1
    assert citations[0].chunk_id == "CHUNK_01"

    # Fabricated citation ID
    invalid_text = "Metformin cures heart disease [Source: NON_EXISTENT_CHUNK_999]."
    is_valid, citations, warnings = validator.validate_citations(invalid_text, evidence)
    assert is_valid is False
    assert any("FABRICATED_OR_MISSING_CITATION_ID" in w for w in warnings)


# 19. Missing Evidence & 20. Unknown Medicine Handling
def test_missing_evidence_and_unknown_medicine_handling():
    llm = MedicalLLM()
    safety = SafetyAssessment(safety_level="LOW", is_emergency=False, is_safe_to_generate=True)
    resp = llm.generate_response(
        query="What is the dose for Fakedrugomine?",
        intent="DOSAGE_INFORMATION",
        safety=safety,
        entities=[MedicalEntity(text="Fakedrugomine", label="DRUG", start_char=0, end_char=13)],
        evidence=[],
        interactions=[]
    )
    assert "not available in the verified clinical monograph database" in resp


# 21. Unverified Prescription Medicine vs 22. Verified Prescription Medicine
def test_prescription_context_verification_handling():
    llm = MedicalLLM()
    safety = SafetyAssessment(safety_level="LOW", is_emergency=False, is_safe_to_generate=True)
    
    # Unverified / review required context
    unverified_ctx = {
        "candidates": [
            {"raw_text": "met...", "verification_status": "review_required", "confidence": 0.40}
        ]
    }
    resp_unverified = llm.generate_response(
        query="Explain my prescription",
        intent="PRESCRIPTION",
        safety=safety,
        entities=[],
        evidence=[],
        interactions=[],
        prescription_context=unverified_ctx
    )
    assert "could not be reliably verified" in resp_unverified
    assert "inspect the original prescription before taking any medication" in resp_unverified

    # Verified context
    verified_ctx = {
        "candidates": [
            {"raw_text": "Amoxicillin 500mg", "normalized_name": "Amoxicillin", "verification_status": "verified", "confidence": 0.98}
        ]
    }
    resp_verified = llm.generate_response(
        query="Explain my prescription",
        intent="PRESCRIPTION",
        safety=safety,
        entities=[],
        evidence=[],
        interactions=[],
        prescription_context=verified_ctx
    )
    assert "Found confirmed medication(s): Amoxicillin" in resp_verified


# 23. Drug-Interaction Evidence Lookup & 24. Absent Interaction Evidence
def test_drug_interaction_engine():
    engine = DrugInteractionEngine()

    # Known major interaction: Warfarin + Aspirin
    res_major = engine.check_pair_interaction("Warfarin", "Aspirin")
    assert res_major.interaction_found is True
    assert res_major.severity == "MAJOR"
    assert "bleeding" in res_major.clinical_effect.lower()
    assert "DailyMed" in res_major.evidence_source

    # Absent interaction: Paracetamol + Cetirizine (unverified pair)
    res_absent = engine.check_pair_interaction("Paracetamol", "Cetirizine")
    assert res_absent.interaction_found is False
    assert res_absent.severity is None
    assert "No verified interaction documented" in res_absent.clinical_effect


# 25. LLM Prompt Construction
def test_llm_prompt_construction():
    llm = MedicalLLM()
    prompt = llm.build_system_prompt()
    assert "Answer ONLY using the facts presented" in prompt
    assert "Do NOT invent, assume, or extrapolate" in prompt
    assert "[Source: chunk_id]" in prompt


# 26. Evaluator Metrics Framework
def test_rag_evaluator_metrics():
    evaluator = MedicalRAGEvaluator()
    # MRR
    mrr = evaluator.compute_mrr([["doc1", "doc2"], ["doc3", "doc1"]], ["doc1", "doc1"])
    assert mrr == 0.75  # (1/1 + 1/2) / 2

    # Safety metrics
    preds = ["EMERGENCY", "LOW", "EMERGENCY", "LOW"]
    truths = ["EMERGENCY", "LOW", "LOW", "EMERGENCY"]
    metrics = evaluator.evaluate_safety_classifier(preds, truths)
    assert "emergency_precision" in metrics
    assert "emergency_recall" in metrics
    assert "false_negative_rate" in metrics
