"""Master Orchestrator for Phase 10B Medical Embedding & Retrieval Upgrade Benchmark."""
import asyncio
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

proj_root = Path(__file__).resolve().parent.parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from evaluation.analysis.retrieval_failure_analysis import run_full_failure_analysis
from evaluation.benchmarks.embedding_retrieval_benchmark import run_embedding_benchmark
from evaluation.benchmarks.hybrid_retrieval_benchmark import run_hybrid_retrieval_benchmark
from evaluation.benchmarks.reranking_benchmark import run_reranking_benchmark
from evaluation.utilities.helpers import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("medicine_ai.phase10b.master")


def generate_comparison_csv(
    dense_metrics: Dict[str, Any],
    hybrid_metrics: Dict[str, Any],
    reranked_metrics: Dict[str, Any],
    csv_path: Path
) -> None:
    """Generate structured Phase 10B comparison CSV table."""
    headers = [
        "System",
        "Recall@5",
        "Recall@10",
        "MRR",
        "nDCG@5"
    ]

    rows = [
        # Lexical
        ["Lexical", hybrid_metrics["lexical"]["recall_at_5"], hybrid_metrics["lexical"]["recall_at_10"], hybrid_metrics["lexical"]["mrr"], hybrid_metrics["lexical"]["ndcg_at_5"]],
        # Current
        ["Current Dense", dense_metrics["current"]["recall_at_5"], dense_metrics["current"]["recall_at_10"], dense_metrics["current"]["mrr"], dense_metrics["current"]["ndcg_at_5"]],
        ["Current Hybrid", hybrid_metrics["current_hybrid"]["recall_at_5"], hybrid_metrics["current_hybrid"]["recall_at_10"], hybrid_metrics["current_hybrid"]["mrr"], hybrid_metrics["current_hybrid"]["ndcg_at_5"]],
        ["Current Hybrid + Reranker", reranked_metrics["current_hybrid_reranked"]["recall_at_5"], "-", reranked_metrics["current_hybrid_reranked"]["mrr"], reranked_metrics["current_hybrid_reranked"]["ndcg_at_5"]],
        # SapBERT
        ["SapBERT Dense", dense_metrics["sapbert"]["recall_at_5"], dense_metrics["sapbert"]["recall_at_10"], dense_metrics["sapbert"]["mrr"], dense_metrics["sapbert"]["ndcg_at_5"]],
        ["SapBERT Hybrid", hybrid_metrics["sapbert_hybrid"]["recall_at_5"], hybrid_metrics["sapbert_hybrid"]["recall_at_10"], hybrid_metrics["sapbert_hybrid"]["mrr"], hybrid_metrics["sapbert_hybrid"]["ndcg_at_5"]],
        ["SapBERT Hybrid + Reranker", reranked_metrics["sapbert_hybrid_reranked"]["recall_at_5"], "-", reranked_metrics["sapbert_hybrid_reranked"]["mrr"], reranked_metrics["sapbert_hybrid_reranked"]["ndcg_at_5"]],
        # BioLinkBERT
        ["BioLinkBERT Dense", dense_metrics["biolinkbert"]["recall_at_5"], dense_metrics["biolinkbert"]["recall_at_10"], dense_metrics["biolinkbert"]["mrr"], dense_metrics["biolinkbert"]["ndcg_at_5"]],
        ["BioLinkBERT Hybrid", hybrid_metrics["biolinkbert_hybrid"]["recall_at_5"], hybrid_metrics["biolinkbert_hybrid"]["recall_at_10"], hybrid_metrics["biolinkbert_hybrid"]["mrr"], hybrid_metrics["biolinkbert_hybrid"]["ndcg_at_5"]],
        ["BioLinkBERT Hybrid + Reranker", reranked_metrics["biolinkbert_hybrid_reranked"]["recall_at_5"], "-", reranked_metrics["biolinkbert_hybrid_reranked"]["mrr"], reranked_metrics["biolinkbert_hybrid_reranked"]["ndcg_at_5"]],
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    logger.info(f"Comparison CSV saved to {csv_path}")


async def execute_phase10b_pipeline():
    """Run all Phase 10B benchmarks end-to-end and generate reports."""
    chunks_path = proj_root / "data" / "processed" / "knowledge_base" / "clinical_chunks.json"
    eval_dataset_path = proj_root / "evaluation" / "datasets" / "rag_eval.jsonl"
    indexes_dir = proj_root / "data" / "indexes"
    output_dir = proj_root / "reports" / "evaluation"

    logger.info("Step 1: Running Dense Embedding Retrieval Benchmark...")
    dense_out = run_embedding_benchmark(chunks_path, eval_dataset_path, indexes_dir, output_dir)

    logger.info("Step 2: Running Hybrid Retrieval Fusion Benchmark...")
    hybrid_out = run_hybrid_retrieval_benchmark(chunks_path, eval_dataset_path, dense_out, output_dir)

    logger.info("Step 3: Running Cross-Encoder Reranker Benchmark...")
    rerank_out = await run_reranking_benchmark(chunks_path, eval_dataset_path, hybrid_out["systems"], output_dir)

    # Combine all system rankings for failure analysis
    combined_rankings = dict(hybrid_out["systems"])
    for k, v in rerank_out["reranked_metrics"].items():
        combined_rankings[k] = v["rankings"]

    logger.info("Step 4: Running Failure Taxonomy & Category Analysis...")
    run_full_failure_analysis(chunks_path, eval_dataset_path, combined_rankings, output_dir)

    logger.info("Step 5: Generating Comparison CSV...")
    generate_comparison_csv(
        dense_out["results"],
        hybrid_out["metrics"],
        rerank_out["reranked_metrics"],
        output_dir / "phase10b_comparison.csv"
    )

    logger.info("=" * 70)
    logger.info("PHASE 10B BENCHMARK SUITE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(execute_phase10b_pipeline())
