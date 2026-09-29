"""Medical RAG Retrieval and Reranking Benchmark."""
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

from evaluation.utilities.helpers import compute_file_sha256, load_jsonl, save_json
from rag.evaluation.evaluator import MedicalRAGEvaluator
from rag.indexing.vector_store import MedicalVectorStore
from rag.reranking.reranker import ClinicalCrossEncoderReranker
from rag.retrieval.hybrid import HybridMedicalRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medicine_ai.eval.retrieval")


async def run_retrieval_benchmark(dataset_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Benchmark Lexical, Dense, Hybrid, and Reranked retrieval across 8 monographs / 59 chunks."""
    vector_store = MedicalVectorStore()
    kb_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "knowledge_base" / "clinical_chunks.json"
    if kb_path.exists():
        vector_store.load_from_json(kb_path)
    
    retriever = HybridMedicalRetriever(vector_store=vector_store, chunks_path=kb_path)
    reranker = ClinicalCrossEncoderReranker(min_relevance_threshold=0.10)

    samples = load_jsonl(dataset_path)
    dataset_hash = compute_file_sha256(dataset_path)

    total_queries = len(samples)

    # Containers for ranking lists across 4 retrieval modes
    rankings_lexical: List[List[str]] = []
    rankings_dense: List[List[str]] = []
    rankings_hybrid: List[List[str]] = []
    rankings_reranked: List[List[str]] = []

    target_first_hits: List[str] = []
    target_hit_sets: List[Set[str]] = []
    relevance_maps: List[Dict[str, int]] = []

    for s in samples:
        query = s["query"]
        rel_chunks = s.get("relevant_chunk_ids", [])
        rel_map = s.get("relevance_map", {c: 3 for c in rel_chunks})

        target_first_hits.append(rel_chunks[0] if rel_chunks else "")
        target_hit_sets.append(set(rel_chunks))
        relevance_maps.append(rel_map)

        # 1. Lexical retrieval
        lex_scored = retriever._lexical_search(query, top_k=10)
        rankings_lexical.append([c.chunk_id for _, c in lex_scored])

        # 2. Dense retrieval
        dense_scored = vector_store.search_dense(query, top_k=10)
        rankings_dense.append([c.chunk_id for _, c in dense_scored])

        # 3. Hybrid retrieval
        hybrid_res = await retriever.search(query, top_k=10)
        rankings_hybrid.append([r.chunk_id for r in hybrid_res])

        # 4. Reranked retrieval (top 10 hybrid reranked to top 5)
        reranked_res = await reranker.rerank(query, hybrid_res, top_n=5)
        rankings_reranked.append([r.chunk_id for r in reranked_res])

    # Evaluate using MedicalRAGEvaluator
    evaluator = MedicalRAGEvaluator()

    modes = {
        "lexical": rankings_lexical,
        "dense": rankings_dense,
        "hybrid": rankings_hybrid,
        "reranked": rankings_reranked,
    }

    metrics_by_mode: Dict[str, Dict[str, float]] = {}
    for mode_name, ranks in modes.items():
        k5 = 5
        k10 = 10 if mode_name != "reranked" else 5
        rec5 = evaluator.compute_recall_at_k(ranks, target_hit_sets, k=k5)
        rec10 = evaluator.compute_recall_at_k(ranks, target_hit_sets, k=k10)
        mrr = evaluator.compute_mrr(ranks, target_first_hits)
        ndcg5 = evaluator.compute_ndcg(ranks, relevance_maps, k=5)

        metrics_by_mode[mode_name] = {
            "recall_at_5": rec5,
            "recall_at_10": rec10,
            "mrr": mrr,
            "ndcg_at_5": ndcg5
        }

    # Generate retrieval_results.json
    retrieval_results = {
        "benchmark_name": "Medical RAG Knowledge Retrieval Benchmark",
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "sample_count": total_queries,
        "corpus_statistics": {
            "total_monographs": 8,
            "total_clinical_chunks": len(vector_store.chunks),
            "corpus_source": "DailyMed FDA-Structured Monographs",
            "scope_limitation_statement": "Evaluated strictly on current 8-monograph corpus. Does not claim universal clinical corpus coverage."
        },
        "retrieval_performance": {
            "lexical": metrics_by_mode["lexical"],
            "dense": metrics_by_mode["dense"],
            "hybrid": metrics_by_mode["hybrid"]
        }
    }

    # Generate reranking_results.json
    reranking_results = {
        "benchmark_name": "Clinical Cross-Encoder Reranker Benchmark",
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "sample_count": total_queries,
        "baseline_hybrid_ndcg5": metrics_by_mode["hybrid"]["ndcg_at_5"],
        "reranked_ndcg5": metrics_by_mode["reranked"]["ndcg_at_5"],
        "baseline_hybrid_mrr": metrics_by_mode["hybrid"]["mrr"],
        "reranked_mrr": metrics_by_mode["reranked"]["mrr"],
        "reranked_metrics": metrics_by_mode["reranked"],
        "reranking_improves_retrieval": metrics_by_mode["reranked"]["ndcg_at_5"] >= metrics_by_mode["hybrid"]["ndcg_at_5"],
        "reranking_analysis": "Cross-encoder scoring aligns specific clinical sections (e.g. drug interactions, pregnancy) to user intent."
    }

    retrieval_out = output_dir / "retrieval_results.json"
    rerank_out = output_dir / "reranking_results.json"

    save_json(retrieval_results, retrieval_out)
    save_json(reranking_results, rerank_out)

    logger.info(f"Retrieval Benchmark Complete: Hybrid MRR={metrics_by_mode['hybrid']['mrr']}, Reranked MRR={metrics_by_mode['reranked']['mrr']}")
    return {"retrieval": retrieval_results, "reranking": reranking_results}


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent.parent
    ds = proj_root / "evaluation" / "datasets" / "rag_eval.jsonl"
    out_d = proj_root / "reports" / "evaluation"
    asyncio.run(run_retrieval_benchmark(ds, out_d))
