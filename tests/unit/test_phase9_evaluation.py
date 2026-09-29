"""Unit and integration tests for Phase 9 Comprehensive Evaluation & Clinical Safety Validation."""
import asyncio
import json
from pathlib import Path
import pytest

from evaluation.benchmarks.e2e_benchmark import run_e2e_benchmark
from evaluation.benchmarks.generation_benchmark import run_generation_benchmark
from evaluation.benchmarks.handwriting_benchmark import run_handwriting_benchmark
from evaluation.benchmarks.interaction_benchmark import run_interaction_benchmark
from evaluation.benchmarks.ner_benchmark import run_ner_benchmark
from evaluation.benchmarks.normalization_benchmark import run_normalization_benchmark
from evaluation.benchmarks.ocr_benchmark import run_ocr_benchmark
from evaluation.benchmarks.performance_benchmark import run_performance_benchmark
from evaluation.benchmarks.retrieval_benchmark import run_retrieval_benchmark
from evaluation.benchmarks.safety_benchmark import run_safety_benchmark
from evaluation.benchmarks.security_benchmark import run_security_benchmark
from evaluation.utilities.helpers import compute_file_sha256, load_jsonl


@pytest.fixture
def proj_root():
    return Path(__file__).resolve().parent.parent.parent


def test_evaluation_datasets_exist_and_valid_jsonl(proj_root):
    """Verify that all 6 evaluation datasets exist, have non-empty records, and calculate SHA256."""
    ds_dir = proj_root / "evaluation" / "datasets"
    expected_files = [
        "prescription_eval.jsonl",
        "handwriting_eval.jsonl",
        "normalization_eval.jsonl",
        "rag_eval.jsonl",
        "safety_eval.jsonl",
        "interaction_eval.jsonl"
    ]
    for fname in expected_files:
        fpath = ds_dir / fname
        assert fpath.exists(), f"Missing evaluation dataset {fname}"
        records = load_jsonl(fpath)
        assert len(records) > 0, f"Dataset {fname} is empty"
        h = compute_file_sha256(fpath)
        assert len(h) == 64, f"Invalid SHA-256 hash length for {fname}"


def test_ocr_benchmark_execution(proj_root, tmp_path):
    """Verify prescription OCR benchmark computes CER and WER correctly."""
    ds_path = proj_root / "evaluation" / "datasets" / "prescription_eval.jsonl"
    out_path = tmp_path / "ocr_test_results.json"
    results = run_ocr_benchmark(ds_path, out_path)

    assert out_path.exists()
    assert "average_cer" in results
    assert "average_wer" in results
    assert results["sample_count"] == 8
    assert results["exact_transcription_accuracy"] >= 0.0


def test_handwriting_benchmark_execution(proj_root, tmp_path):
    """Verify handwriting benchmark evaluates Doctor BD classes and Top-1/Top-5 accuracy."""
    ds_path = proj_root / "evaluation" / "datasets" / "handwriting_eval.jsonl"
    out_path = tmp_path / "handwriting_test_results.json"
    results = run_handwriting_benchmark(ds_path, out_path)

    assert out_path.exists()
    assert "top1_accuracy" in results
    assert "top5_accuracy" in results
    assert results["known_cross_split_duplicates_documented"] == 55
    assert results["sample_count"] == 40


def test_ner_benchmark_execution(proj_root, tmp_path):
    """Verify clinical NER benchmark evaluates multi-class entity extraction."""
    out_path = tmp_path / "ner_test_results.json"
    results = run_ner_benchmark(out_path)

    assert out_path.exists()
    assert results["overall_f1_score"] >= 0.80
    assert results["overall_recall"] >= 0.90
    assert "DRUG" in results["per_label_metrics"]


def test_normalization_benchmark_execution(proj_root, tmp_path):
    """Verify multi-tier normalization benchmark handles exact, alias, and OOV cases."""
    ds_path = proj_root / "evaluation" / "datasets" / "normalization_eval.jsonl"
    out_path = tmp_path / "normalization_test_results.json"
    results = run_normalization_benchmark(ds_path, out_path)

    assert out_path.exists()
    assert results["top1_accuracy"] >= 0.85
    assert results["exact_rxcui_match_rate"] >= 0.90
    assert results["sample_count"] == 20


@pytest.mark.asyncio
async def test_retrieval_and_reranking_benchmark(proj_root, tmp_path):
    """Verify retrieval & cross-encoder reranking benchmark across 4 modes."""
    ds_path = proj_root / "evaluation" / "datasets" / "rag_eval.jsonl"
    res = await run_retrieval_benchmark(ds_path, tmp_path)

    assert (tmp_path / "retrieval_results.json").exists()
    assert (tmp_path / "reranking_results.json").exists()
    assert "reranked_metrics" in res["reranking"]
    assert res["reranking"]["reranked_ndcg5"] >= 0.70


@pytest.mark.asyncio
async def test_generation_groundedness_benchmark(proj_root, tmp_path):
    """Verify grounded generation benchmark checks citations and hallucination rate."""
    out_path = tmp_path / "generation_test_results.json"
    results = await run_generation_benchmark(out_path)

    assert out_path.exists()
    assert results["citation_correctness_rate"] == 1.0
    assert results["hallucination_rate"] == 0.0
    assert results["evidence_grounding_rate"] == 1.0


def test_drug_interaction_benchmark(proj_root, tmp_path):
    """Verify drug interaction evaluation preserves symmetry and handles unknown pairs."""
    ds_path = proj_root / "evaluation" / "datasets" / "interaction_eval.jsonl"
    out_path = tmp_path / "interaction_test_results.json"
    results = run_interaction_benchmark(ds_path, out_path)

    assert out_path.exists()
    assert results["interaction_detection_accuracy"] == 1.0
    assert results["pair_symmetry_preserved"] is True
    assert results["llm_reasoning_used_as_substitute"] is False


def test_safety_benchmark_emergency_sensitivity(proj_root, tmp_path):
    """Verify clinical safety guardrails achieve 0 emergency false negatives."""
    ds_path = proj_root / "evaluation" / "datasets" / "safety_eval.jsonl"
    out_path = tmp_path / "safety_test_results.json"
    results = run_safety_benchmark(ds_path, out_path)

    assert out_path.exists()
    assert results["emergency_detection_metrics"]["false_negatives"] == 0
    assert results["emergency_detection_metrics"]["recall"] == 1.0
    assert results["emergency_false_negatives_count"] == 0


@pytest.mark.asyncio
async def test_e2e_pipeline_benchmark(proj_root, tmp_path):
    """Verify end-to-end pipeline benchmark executes all stages."""
    ds_path = proj_root / "evaluation" / "datasets" / "prescription_eval.jsonl"
    out_path = tmp_path / "e2e_test_results.json"
    results = await run_e2e_benchmark(ds_path, out_path)

    assert out_path.exists()
    assert results["metrics"]["success_rate"] == 1.0
    assert results["metrics"]["grounded_response_rate"] == 1.0
    assert results["latency_profile_ms"]["average_total_pipeline_ms"] > 0.0


def test_security_benchmark_execution(proj_root, tmp_path):
    """Verify security benchmark audits secrets, upload guards, and citation spoofing."""
    out_path = tmp_path / "security_test_results.json"
    results = run_security_benchmark(out_path)

    assert out_path.exists()
    assert results["security_verifications"]["frontend_secrets_clean"] is True
    assert results["security_verifications"]["raw_prescription_disk_persistence"] is False
    assert results["security_verifications"]["upload_security_tests"]["passed"] == 4
