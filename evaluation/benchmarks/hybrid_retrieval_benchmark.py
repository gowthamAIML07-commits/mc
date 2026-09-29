"""Phase 10B: Hybrid Medical Knowledge Retrieval Benchmark (RRF Fusion).

Evaluates Lexical, Dense, and Hybrid Reciprocal Rank Fusion across:
1. Lexical Baseline
2. Current Dense & Current Hybrid
3. SapBERT Dense & SapBERT Hybrid
4. BioLinkBERT Dense & BioLinkBERT Hybrid
"""
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import numpy as np

proj_root = Path(__file__).resolve().parent.parent.parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from evaluation.utilities.helpers import compute_file_sha256, load_jsonl, save_json
from rag.evaluation.evaluator import MedicalRAGEvaluator
from rag.retrieval.hybrid import HybridMedicalRetriever
from rag.schemas import DocumentChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medicine_ai.eval.hybrid_benchmark")


def compute_rrf_hybrid_rankings(
    lexical_rankings: List[List[Tuple[float, str]]],
    dense_rankings: List[List[str]],
    rrf_k: int = 60,
    top_k: int = 10
) -> List[List[str]]:
    """Compute Reciprocal Rank Fusion rankings between lexical and dense ranking lists."""
    fused_rankings: List[List[str]] = []

    for lex_list, dense_list in zip(lexical_rankings, dense_rankings):
        scores: Dict[str, float] = {}

        # 1. Lexical contributions
        for rank, (score, chunk_id) in enumerate(lex_list, 1):
            lex_weight = 3.0 if score >= 0.4 else 1.5
            scores[chunk_id] = scores.get(chunk_id, 0.0) + lex_weight * (1.0 / (rrf_k + rank)) + score

        # 2. Dense contributions
        for rank, chunk_id in enumerate(dense_list, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 * (1.0 / (rrf_k + rank))

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused_rankings.append([cid for cid, _ in sorted_items[:top_k]])

    return fused_rankings


def run_hybrid_retrieval_benchmark(
    chunks_path: Path,
    eval_dataset_path: Path,
    dense_results: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """Execute hybrid retrieval benchmark across all models."""
    logger.info("=" * 70)
    logger.info("Executing Phase 10B Hybrid Retrieval Benchmark...")
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(chunks_path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)
    chunks = [DocumentChunk(**c) for c in raw_chunks]

    eval_samples = load_jsonl(eval_dataset_path)
    queries = [s["query"] for s in eval_samples]
    target_first_hits = [s.get("relevant_chunk_ids", [""])[0] for s in eval_samples]
    target_hit_sets = [set(s.get("relevant_chunk_ids", [])) for s in eval_samples]
    relevance_maps = [s.get("relevance_map", {c: 3 for c in s.get("relevant_chunk_ids", [])}) for s in eval_samples]

    evaluator = MedicalRAGEvaluator()
    retriever = HybridMedicalRetriever(chunks_path=chunks_path)

    # 1. Compute Lexical baseline
    lexical_raw = []
    lexical_rankings = []
    for q in queries:
        lex_scored = retriever._lexical_search(q, top_k=20)
        lexical_raw.append([(score, c.chunk_id) for score, c in lex_scored])
        lexical_rankings.append([c.chunk_id for _, c in lex_scored])

    # 2. Build hybrid rankings for each model
    systems = {
        "lexical": lexical_rankings,
        "current_dense": dense_results["results"]["current"]["rankings"],
        "current_hybrid": compute_rrf_hybrid_rankings(
            lexical_raw,
            dense_results["results"]["current"]["rankings"]
        ),
        "sapbert_dense": dense_results["results"]["sapbert"]["rankings"],
        "sapbert_hybrid": compute_rrf_hybrid_rankings(
            lexical_raw,
            dense_results["results"]["sapbert"]["rankings"]
        ),
        "biolinkbert_dense": dense_results["results"]["biolinkbert"]["rankings"],
        "biolinkbert_hybrid": compute_rrf_hybrid_rankings(
            lexical_raw,
            dense_results["results"]["biolinkbert"]["rankings"]
        )
    }

    metrics_table = {}
    for system_name, ranks in systems.items():
        rec5 = evaluator.compute_recall_at_k(ranks, target_hit_sets, k=5)
        rec10 = evaluator.compute_recall_at_k(ranks, target_hit_sets, k=10)
        mrr = evaluator.compute_mrr(ranks, target_first_hits)
        ndcg5 = evaluator.compute_ndcg(ranks, relevance_maps, k=5)
        ndcg10 = evaluator.compute_ndcg(ranks, relevance_maps, k=10)

        metrics_table[system_name] = {
            "recall_at_5": rec5,
            "recall_at_10": rec10,
            "mrr": mrr,
            "ndcg_at_5": ndcg5,
            "ndcg_at_10": ndcg10
        }
        logger.info(f"[{system_name.upper()}] Recall@5={rec5:.4f}, MRR={mrr:.4f}, nDCG@5={ndcg5:.4f}")

    results_payload = {
        "benchmark_name": "Phase 10B Hybrid Retrieval Fusion Benchmark",
        "dataset_path": str(eval_dataset_path),
        "dataset_sha256": compute_file_sha256(eval_dataset_path),
        "corpus_sha256": compute_file_sha256(chunks_path),
        "query_count": len(queries),
        "rrf_parameter_k": 60,
        "metrics": metrics_table,
        "rankings": systems
    }

    save_json(results_payload, output_dir / "phase10b_retrieval_results.json")
    return {"metrics": metrics_table, "systems": systems, "lexical_raw": lexical_raw}
