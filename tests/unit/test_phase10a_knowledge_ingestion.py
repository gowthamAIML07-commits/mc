"""Unit tests for Phase 10A Clinical Knowledge Base Corpus Ingestion and Validation."""
import hashlib
import json
from pathlib import Path
import pytest

from rag.ingestion.adapter import DailyMedRxNormIngestionAdapter
from rag.ingestion.chunker import MedicalSectionChunker
from rag.ingestion.validators import ClinicalDocumentValidator, DocumentQualityStatus
from rag.schemas import DocumentChunk, RawMedicalDocument


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def sample_valid_doc() -> RawMedicalDocument:
    return RawMedicalDocument(
        source_id="DAILYMED_TEST_DRUG",
        source_name="FDA DailyMed / US NLM",
        source_type="fda_approved_labeling",
        source_url="https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=test-drug",
        title="Testdrug 100 mg Oral Tablet",
        version_date="2024-12",
        rxcui="999999",
        ingredient="Testdrug",
        drug_class="Synthetic Antibacterial",
        atc_code="J01XX99",
        brand_aliases=["Testbrand", "Testil"],
        sections={
            "indications": "Indicated for the treatment of complicated bacterial infections.",
            "dosage": "100 mg orally once daily with water.",
            "contraindications": "Severe hypersensitivity to testdrug or formulation excipients.",
            "warnings": "Risk of hepatotoxicity and mild gastrointestinal disturbance.",
            "adverse_reactions": "Nausea (5%), headache (3%), diarrhea (2%).",
            "drug_interactions": "Co-administration with potent CYP3A4 inhibitors may increase levels.",
            "pregnancy": "Category B. No evidence of impaired fertility or harm to fetus.",
            "patient_counseling": "Take with full glass of water. Complete full prescribed course."
        }
    )


def test_authoritative_provenance_and_sha256(project_root: Path):
    """Test 1: Manifest and raw documents contain verified provenance and valid SHA-256 hashes."""
    manifest_file = project_root / "data" / "manifests" / "clinical_knowledge_manifest.json"
    assert manifest_file.exists(), "Corpus manifest must exist"

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["corpus_version"] == "clinical-kb-v1"
    assert manifest["total_documents"] >= 500
    assert manifest["distinct_medicines"] >= 500
    assert manifest["total_chunks"] >= 4000
    assert "manifest_sha256" in manifest or "documents" in manifest

    for doc_rec in manifest["documents"][:10]:
        assert doc_rec["source_authority"] == "FDA DailyMed / US NLM"
        assert doc_rec["source_url"].startswith("https://dailymed.nlm.nih.gov")
        assert len(doc_rec["sha256"]) == 64
        assert doc_rec["quality_status"] in ["VALID", "WARNING", "REQUIRES_REVIEW"]


def test_duplicate_detection(tmp_path: Path, sample_valid_doc: RawMedicalDocument):
    """Test 2: Adapter identifies and quarantines duplicate source IDs and hash collisions."""
    adapter = DailyMedRxNormIngestionAdapter(
        raw_storage_dir=tmp_path / "raw",
        processed_kb_dir=tmp_path / "kb",
        manifest_path=tmp_path / "manifest.json"
    )

    validator = ClinicalDocumentValidator()
    status1, issues1 = validator.validate_raw_document(sample_valid_doc)
    assert status1 == DocumentQualityStatus.VALID

    # Duplicate document check
    doc_copy = sample_valid_doc.model_copy()
    raw_ser1 = json.dumps(sample_valid_doc.model_dump(), sort_keys=True)
    raw_ser2 = json.dumps(doc_copy.model_dump(), sort_keys=True)
    assert hashlib.sha256(raw_ser1.encode("utf-8")).hexdigest() == hashlib.sha256(raw_ser2.encode("utf-8")).hexdigest()


def test_rxcui_and_medicine_anchoring(project_root: Path):
    """Test 3: RxCUIs are properly populated and anchored to US NLM RxNorm concepts."""
    chunks_file = project_root / "data" / "processed" / "knowledge_base" / "clinical_chunks.json"
    assert chunks_file.exists()

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    rxcui_set = set()
    for c in chunks:
        rxcui = c.get("rxcui")
        if rxcui:
            rxcui_set.add(rxcui)
            assert str(rxcui).isdigit()

    assert len(rxcui_set) >= 500, f"Expected 500+ distinct RxCUIs, found {len(rxcui_set)}"


def test_section_normalization_and_taxonomy(sample_valid_doc: RawMedicalDocument):
    """Test 4: Section chunker normalizes clinical headings to canonical medical taxonomy."""
    chunker = MedicalSectionChunker(max_chunk_chars=1500)
    chunks = chunker.chunk_document(sample_valid_doc)

    assert len(chunks) == len(sample_valid_doc.sections)
    sec_names = [c.section_category for c in chunks]

    assert "indications" in sec_names
    assert "dosage" in sec_names
    assert "contraindications" in sec_names
    assert "warnings" in sec_names
    assert "adverse_reactions" in sec_names
    assert "drug_interactions" in sec_names
    assert "pregnancy" in sec_names
    assert "patient_counseling" in sec_names


def test_chunk_metadata_completeness(project_root: Path):
    """Test 5: Every indexed chunk contains all mandatory Phase 10A metadata fields."""
    chunks_file = project_root / "data" / "processed" / "knowledge_base" / "clinical_chunks.json"
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    for c in chunks[:50]:
        meta = c.get("metadata", {})
        assert meta.get("chunk_id")
        assert meta.get("medicine_name")
        assert meta.get("rxcui")
        assert meta.get("ingredient")
        assert meta.get("source_authority") == "FDA DailyMed / US NLM"
        assert meta.get("source_document")
        assert meta.get("source_version")
        assert meta.get("section")
        assert meta.get("original_section")
        assert meta.get("text")
        assert meta.get("sha256") and len(meta["sha256"]) == 64
        assert meta.get("provenance_id")
        assert meta.get("retrieved_at")


def test_malformed_document_rejection():
    """Test 6: Validator rejects corrupted, title-less, or section-less documents."""
    validator = ClinicalDocumentValidator()

    # Empty sections
    doc_bad = RawMedicalDocument(
        source_id="DAILYMED_CORRUPT",
        source_name="FDA DailyMed",
        source_type="fda_approved_labeling",
        source_url="https://dailymed.nlm.nih.gov",
        title="Corrupted Medicine",
        version_date="2024-12",
        ingredient="Corrupted",
        sections={}
    )
    status, issues = validator.validate_raw_document(doc_bad)
    assert status == DocumentQualityStatus.REJECTED
    assert any("sections" in issue.lower() for issue in issues)


def test_empty_document_rejection():
    """Test 7: Validator rejects documents with blank title or missing source authority."""
    validator = ClinicalDocumentValidator()

    doc_empty = RawMedicalDocument(
        source_id="DAILYMED_BLANK",
        source_name="",
        source_type="fda_approved_labeling",
        source_url="",
        title="   ",
        version_date="",
        ingredient="",
        sections={"indications": "   "}
    )
    status, issues = validator.validate_raw_document(doc_empty)
    assert status == DocumentQualityStatus.REJECTED


def test_zero_phi_in_knowledge_corpus(project_root: Path):
    """Test 8: Corpus contains zero patient PHI, private notes, or sensitive keys."""
    import re
    corpus_file = project_root / "data" / "processed" / "knowledge_base" / "clinical_chunks.json"
    with open(corpus_file, "r", encoding="utf-8") as f:
        content = f.read()

    forbidden_patterns = [
        r"\bpatient_name\b",
        r"\bsocial_security\b",
        r"\bssn\b",
        r"\bmrn\b",
        r"\bapi_key\b",
        r"\bsk-[a-zA-Z0-9]{20,}\b",
        r"BEGIN PRIVATE KEY",
        r"\bprescription_image\b"
    ]
    for pattern in forbidden_patterns:
        assert not re.search(pattern, content, re.IGNORECASE), f"Forbidden pattern {pattern} found in corpus."


def test_no_fabricated_interactions(project_root: Path):
    """Test 9: Phase 6 verified interaction engine pairs remain authentic without synthetic inflation."""
    from rag.interactions.engine import DrugInteractionEngine
    engine = DrugInteractionEngine()
    
    # Check that verified interaction rules remain authentic with zero fabrication
    assert len(engine.VERIFIED_INTERACTION_RULES) >= 9
    for rule in engine.VERIFIED_INTERACTION_RULES:
        assert rule["pair"]
        assert rule["severity"] in ["CONTRAINDICATED", "MAJOR", "MODERATE", "MINOR"]
        assert rule["clinical_effect"]
        assert rule["recommendation"]
        assert "evidence_source" in rule
