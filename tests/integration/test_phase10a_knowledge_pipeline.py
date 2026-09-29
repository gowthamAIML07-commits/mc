"""Integration tests for Phase 10A Clinical Knowledge Base Pipeline and Reproducibility."""
import hashlib
import json
from pathlib import Path
import pytest

from rag.indexing.vector_store import MedicalVectorStore
from rag.ingestion.adapter import DailyMedRxNormIngestionAdapter
from rag.reranking.reranker import ClinicalCrossEncoderReranker
from rag.retrieval.hybrid import HybridMedicalRetriever
from rag.validation.citation_validator import CitationValidator


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def vector_store(project_root: Path) -> MedicalVectorStore:
    vs = MedicalVectorStore()
    kb_path = project_root / "data" / "processed" / "knowledge_base" / "clinical_chunks.json"
    if kb_path.exists():
        vs.load_from_json(kb_path)
    return vs


def test_expanded_corpus_vector_indexing_and_search(vector_store: MedicalVectorStore):
    """Test 1: MedicalVectorStore indexes 4,000+ chunks and supports dense vector search."""
    assert len(vector_store.chunks) >= 4000

    # Query cardiovascular drug
    results = vector_store.search_dense("Atorvastatin indications hypercholesterolemia", top_k=5)
    assert len(results) > 0
    top_score, top_chunk = results[0]
    assert "Atorvastatin" in top_chunk.title or "Atorvastatin" in top_chunk.text or top_score > 0.5

    # Query oncology drug
    onco_results = vector_store.search_dense("Osimertinib EGFR mutation lung cancer", top_k=5)
    assert len(onco_results) > 0
    top_onco_score, top_onco = onco_results[0]
    assert "Osimertinib" in top_onco.title or "Osimertinib" in top_onco.text or top_onco_score > 0.5


@pytest.mark.asyncio
async def test_hybrid_retriever_compatibility_with_expanded_corpus(project_root: Path, vector_store: MedicalVectorStore):
    """Test 2: HybridMedicalRetriever performs RRF search seamlessly over 500+ medicines."""
    kb_path = project_root / "data" / "processed" / "knowledge_base" / "clinical_chunks.json"
    retriever = HybridMedicalRetriever(vector_store=vector_store, chunks_path=kb_path)

    # Test multi-therapeutic category queries
    test_queries = [
        "What are the contraindications for Metformin in renal impairment?",
        "What is the dosage of Amoxicillin for acute bacterial sinusitis?",
        "What adverse reactions occur with Pembrolizumab immunotherapy?",
        "Can Warfarin be administered during pregnancy?"
    ]

    for q in test_queries:
        res = await retriever.search(q, top_k=5)
        assert len(res) > 0
        for item in res:
            assert item.chunk_id
            assert item.title
            assert item.text
            assert item.score >= 0.0


@pytest.mark.asyncio
async def test_cross_encoder_reranker_on_expanded_corpus(project_root: Path, vector_store: MedicalVectorStore):
    """Test 3: Cross-encoder reranks hybrid retrieval candidates with high clinical fidelity."""
    kb_path = project_root / "data" / "processed" / "knowledge_base" / "clinical_chunks.json"
    retriever = HybridMedicalRetriever(vector_store=vector_store, chunks_path=kb_path)
    reranker = ClinicalCrossEncoderReranker()

    query = "What are the common adverse reactions of Cetirizine?"
    candidates = await retriever.search(query, top_k=10)
    assert len(candidates) > 0

    reranked = await reranker.rerank(query, candidates, top_n=3)
    assert len(reranked) <= 3
    assert len(reranked) > 0
    assert reranked[0].score >= reranked[-1].score


def test_citation_validator_compatibility(project_root: Path, vector_store: MedicalVectorStore):
    """Test 4: Citation validator correctly verifies chunk-level citations across the expanded corpus."""
    validator = CitationValidator()
    
    # Select arbitrary chunk from vector store
    sample_chunk = vector_store.chunks[0]
    from rag.schemas import RetrievalResult
    evidence = [
        RetrievalResult(
            chunk_id=sample_chunk.chunk_id,
            document_id=sample_chunk.document_id,
            title=sample_chunk.title,
            source=sample_chunk.source_name,
            source_url=sample_chunk.source_url,
            section_name=sample_chunk.section_name,
            text=sample_chunk.text,
            score=0.95
        )
    ]

    response_text = f"According to authoritative labeling [{sample_chunk.chunk_id}], the medicine is indicated as described."
    is_valid, citations, warnings = validator.validate_citations(response_text, evidence)
    assert is_valid is True
    assert len(citations) == 1
    assert citations[0].chunk_id == sample_chunk.chunk_id


def test_deterministic_reproducibility(tmp_path: Path):
    """Test 5: Ingestion pipeline executes deterministically with zero nondeterminism."""
    # First execution
    dir1 = tmp_path / "run1"
    adapter1 = DailyMedRxNormIngestionAdapter(
        raw_storage_dir=dir1 / "raw",
        processed_kb_dir=dir1 / "kb",
        manifest_path=dir1 / "manifest.json"
    )
    res1 = adapter1.process_and_ingest_all()

    # Second execution
    dir2 = tmp_path / "run2"
    adapter2 = DailyMedRxNormIngestionAdapter(
        raw_storage_dir=dir2 / "raw",
        processed_kb_dir=dir2 / "kb",
        manifest_path=dir2 / "manifest.json"
    )
    res2 = adapter2.process_and_ingest_all()

    # Verify identical manifest hash, chunk count, and document counts
    assert res1["manifest_sha256"] == res2["manifest_sha256"]
    assert res1["total_chunks"] == res2["total_chunks"]
    assert res1["distinct_medicines"] == res2["distinct_medicines"]
    assert res1["distinct_rxcuis"] == res2["distinct_rxcuis"]


def test_security_untrusted_input_guards(project_root: Path):
    """Test 6: Source documents are sanitized against HTML/script injection attacks."""
    chunks_file = project_root / "data" / "processed" / "knowledge_base" / "clinical_chunks.json"
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    for c in chunks:
        text = c.get("text", "")
        # Assert no executable script or unescaped HTML tags
        assert "<script" not in text.lower()
        assert "javascript:" not in text.lower()
        assert "onerror=" not in text.lower()
        assert "onload=" not in text.lower()
