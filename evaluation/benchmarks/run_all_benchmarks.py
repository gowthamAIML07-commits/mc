"""Master Orchestrator for Phase 9 Comprehensive Evaluation & Clinical Safety Validation."""
import asyncio
import logging
from pathlib import Path

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("medicine_ai.eval.runner")


async def run_all_benchmarks():
    """Execute all 11 evaluation benchmarks and produce all 12 JSON reports."""
    proj_root = Path(__file__).resolve().parent.parent.parent
    ds_dir = proj_root / "evaluation" / "datasets"
    rep_dir = proj_root / "reports" / "evaluation"
    rep_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=================================================================")
    logger.info("STARTING PHASE 9 EVALUATION & CLINICAL SAFETY BENCHMARKS")
    logger.info("=================================================================")

    # 1. OCR Benchmark
    logger.info("--- 1. Running OCR Benchmark ---")
    run_ocr_benchmark(ds_dir / "prescription_eval.jsonl", rep_dir / "ocr_results.json")

    # 2. Handwriting Benchmark
    logger.info("--- 2. Running Handwriting Benchmark ---")
    run_handwriting_benchmark(ds_dir / "handwriting_eval.jsonl", rep_dir / "handwriting_results.json")

    # 3. Clinical NER Benchmark
    logger.info("--- 3. Running Clinical NER Benchmark ---")
    run_ner_benchmark(rep_dir / "ner_results.json")

    # 4. RxNorm Normalization Benchmark
    logger.info("--- 4. Running RxNorm Normalization Benchmark ---")
    run_normalization_benchmark(ds_dir / "normalization_eval.jsonl", rep_dir / "normalization_results.json")

    # 5 & 6. RAG Retrieval & Reranking Benchmarks
    logger.info("--- 5 & 6. Running RAG Retrieval & Reranking Benchmarks ---")
    await run_retrieval_benchmark(ds_dir / "rag_eval.jsonl", rep_dir)

    # 7. Grounded Generation Benchmark
    logger.info("--- 7. Running Grounded Generation Benchmark ---")
    await run_generation_benchmark(rep_dir / "generation_results.json")

    # 8. Drug Interaction Benchmark
    logger.info("--- 8. Running Deterministic Drug Interaction Benchmark ---")
    run_interaction_benchmark(ds_dir / "interaction_eval.jsonl", rep_dir / "interaction_results.json")

    # 9. Clinical Safety & Emergency Benchmark
    logger.info("--- 9. Running Clinical Safety Benchmark ---")
    run_safety_benchmark(ds_dir / "safety_eval.jsonl", rep_dir / "safety_results.json")

    # 10. End-to-End Pipeline Benchmark
    logger.info("--- 10. Running End-to-End Pipeline Benchmark ---")
    await run_e2e_benchmark(ds_dir / "prescription_eval.jsonl", rep_dir / "e2e_results.json")

    # 11. Performance & Latency Benchmark
    logger.info("--- 11. Running Performance & Latency Benchmark ---")
    await run_performance_benchmark(rep_dir / "performance_results.json")

    # 12. Security & Privacy Benchmark
    logger.info("--- 12. Running Security & Privacy Benchmark ---")
    run_security_benchmark(rep_dir / "security_results.json")

    logger.info("=================================================================")
    logger.info("PHASE 9 ALL BENCHMARKS COMPLETED SUCCESSFULLY")
    logger.info("=================================================================")


if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())
