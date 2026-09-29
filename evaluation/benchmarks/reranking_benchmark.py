"""Phase 10B: Clinical Cross-Encoder Reranker Benchmark.

Evaluates how the ClinicalCrossEncoderReranker refines candidate evidence retrieved by:
1. Current Dense -> Reranker
2. Current Hybrid -> Reranker
3. SapBERT Hybrid -> Reranker
4. BioLinkBERT Hybrid -> Reranker
"""
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

proj_root = Path(__file__).resolve().parent.parent.parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from evaluation.utilities.helpers import compute_file_sha256, load_jsonl, save_json
from rag.evaluation.evaluator import MedicalRAGEvaluator
from rag.reranking.reranker import ClinicalCrossEncoderReranker
from rag.schemas import DocumentChunk, RetrievalResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medicine_ai.eval.reranking_benchmark")


async def run_reranking_benchmark(
    chunks_path: Path,
    eval_dataset_path: Path,
    systems_rankings: Dict[str, List[List[str]]],
    output_dir: Path
) -> Dict[str, Any]:
    """Execute cross-encoder reranking over retrieved candidate lists."""
    logger.info("=" * 70)
    logger.info("Executing Phase 10B Cross-Encoder Reranker Benchmark...")
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(chunks_path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)
    chunk_map = {c["chunk_id"]: DocumentChunk(**c) for c in raw_chunks}

    eval_samples = load_jsonl(eval_dataset_path)
    queries = [s["query"] for s in eval_samples]
    target_first_hits = [s.get("relevant_chunk_ids", [""])[0] for s in eval_samples]
    target_hit_sets = [set(s.get("relevant_chunk_ids", [])) for s in eval_samples]
    relevance_maps = [s.get("relevance_map", {c: 3 for c in s.get("relevant_chunk_ids", [])}) for s in eval_samples]

    evaluator = MedicalRAGEvaluator()
    reranker = ClinicalCrossEncoderReranker(min_relevance_threshold=0.10)

    systems_to_rerank = [
        "current_dense",
        "current_hybrid",
        "sapbert_hybrid",
        "biolinkbert_hybrid"
    ]

    reranked_results_table = {}
    comparison_table = {}

    for sys_name in systems_to_rerank:
        candidate_lists = systems_rankings.get(sys_name, [])
        reranked_rankings: List[List[str]] = []

        for q, cand_ids in zip(queries, candidate_lists):
            # Convert candidate chunk IDs to RetrievalResults
            candidate_docs: List[RetrievalResult] = []
            for rank, cid in enumerate(cand_ids[:10], 1):
                if cid in chunk_map:
                    c = chunk_map[cid]
                    candidate_docs.append(RetrievalResult(
                        document_id=c.document_id,
                        chunk_id=c.chunk_id,
                        title=c.title,
                        source=c.source_name,
                        source_url=c.source_url,
                        section_name=c.section_name,
                        text=c.text,
                        score=round(1.0 / rank, 4),
                        retrieval_method="candidate_for_reranking",
                        metadata={
                            "rxcui": c.rxcui,
                            "ingredient": c.ingredient,
                            "brand_aliases": c.brand_aliases,
                            "section_category": c.section_category
                        }
                    ))

            # Execute cross-encoder reranker
            reranked_docs = await reranker.rerank(q, candidate_docs, top_n=5)
            reranked_rankings.append([d.chunk_id for d in reranked_docs if d.chunk_id])

        # Compute post-reranking metrics
        rec5 = evaluator.compute_recall_at_k(reranked_rankings, target_hit_sets, k=5)
        mrr = evaluator.compute_mrr(reranked_rankings, target_first_hits)
        ndcg5 = evaluator.compute_ndcg(reranked_rankings, relevance_maps, k=5)

        reranked_name = f"{sys_name}_reranked"
        reranked_results_table[reranked_name] = {
            "recall_at_5": rec5,
            "mrr": mrr,
            "ndcg_at_5": ndcg5,
            "rankings": reranked_rankings
        }

        # Baseline comparison
        base_ranks = candidate_lists
        base_mrr = evaluator.compute_mrr(base_ranks, target_first_hits)
        base_ndcg5 = evaluator.compute_ndcg(base_ranks, relevance_maps, k=5)

        abs_mrr_diff = round(mrr - base_mrr, 4)
        rel_mrr_diff = round(((mrr - base_mrr) / max(base_mrr, 0.0001)) * 100.0, 2)
        abs_ndcg_diff = round(ndcg5 - base_ndcg5, 4)
        rel_ndcg_diff = round(((ndcg5 - base_ndcg5) / max(base_ndcg5, 0.0001)) * 100.0, 2)

        comparison_table[sys_name] = {
            "baseline_mrr": base_mrr,
            "reranked_mrr": mrr,
            "mrr_absolute_change": abs_mrr_diff,
            "mrr_relative_change_pct": rel_mrr_diff,
            "baseline_ndcg5": base_ndcg5,
            "reranked_ndcg5": ndcg5,
            "ndcg5_absolute_change": abs_ndcg_diff,
            "ndcg5_relative_change_pct": rel_ndcg_diff,
            "reranking_improves_mrr": mrr >= base_mrr,
            "reranking_improves_ndcg": ndcg5 >= base_ndcg5
        }

        logger.info(
            f"[{reranked_name.upper()}] Post-Rerank: MRR={mrr:.4f} ({rel_mrr_diff:+.1f}%), "
            f"nDCG@5={ndcg5:.4f} ({rel_ndcg_diff:+.1f}%)"
        )

    return {
        "reranked_metrics": reranked_results_table,
        "comparisons": comparison_table
    }
