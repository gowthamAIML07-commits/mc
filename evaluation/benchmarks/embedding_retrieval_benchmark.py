"""Phase 10B: Comprehensive Dense Medical Embedding Retrieval Benchmark.

Evaluates Current Embedding vs SapBERT vs BioLinkBERT across:
1. Dense Retrieval Quality (Recall@1, Recall@5, Recall@10, Recall@20, MRR, nDCG@5, nDCG@10)
2. Indexing Performance (Time, Chunks/sec, Index Size, Memory RSS)
3. Query Latencies (p50, p95, p99)
4. Sanity Checks (Norm distribution, NaN/Inf checks)
"""
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

proj_root = Path(__file__).resolve().parent.parent.parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from evaluation.embeddings.benchmark_current import CurrentEmbeddingModel
from evaluation.embeddings.benchmark_sapbert import SapBERTEmbeddingModel
from evaluation.embeddings.benchmark_biolinkbert import BioLinkBERTEmbeddingModel
from evaluation.utilities.helpers import compute_file_sha256, load_jsonl, save_json
from rag.evaluation.evaluator import MedicalRAGEvaluator
from rag.schemas import DocumentChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medicine_ai.eval.dense_benchmark")


def get_process_rss_mb() -> float:
    """Return process RSS memory in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        # Fallback estimation
        return 0.0


def dense_search(
    query_vector: np.ndarray,
    index_vectors: np.ndarray,
    chunks: List[DocumentChunk],
    top_k: int = 20
) -> List[Tuple[float, str]]:
    """Perform cosine dot product search over unit-normalized embeddings."""
    # Dot product of unit vectors = cosine similarity
    similarities = np.dot(index_vectors, query_vector)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(float(similarities[idx]), chunks[idx].chunk_id) for idx in top_indices]


def run_embedding_benchmark(
    chunks_path: Path,
    eval_dataset_path: Path,
    indexes_dir: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """Execute complete dense embedding benchmark across all 3 models."""
    logger.info("=" * 70)
    logger.info("Executing Phase 10B Medical Dense Embedding Benchmark...")
    logger.info("=" * 70)

    indexes_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load clinical chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)
    chunks = [DocumentChunk(**c) for c in raw_chunks]
    chunk_texts = [f"{c.title} - {c.section_name}: {c.text}" for c in chunks]
    corpus_hash = compute_file_sha256(chunks_path)

    # 2. Load evaluation dataset
    eval_samples = load_jsonl(eval_dataset_path)
    eval_hash = compute_file_sha256(eval_dataset_path)
    queries = [s["query"] for s in eval_samples]
    target_first_hits = [s.get("relevant_chunk_ids", [""])[0] for s in eval_samples]
    target_hit_sets = [set(s.get("relevant_chunk_ids", [])) for s in eval_samples]
    relevance_maps = [s.get("relevance_map", {c: 3 for c in s.get("relevant_chunk_ids", [])}) for s in eval_samples]

    evaluator = MedicalRAGEvaluator()

    models_to_test = [
        ("current", CurrentEmbeddingModel),
        ("sapbert", SapBERTEmbeddingModel),
        ("biolinkbert", BioLinkBERTEmbeddingModel)
    ]

    all_model_results = {}
    performance_records = {}

    for model_key, model_cls in models_to_test:
        logger.info(f"\n--- Benchmarking Model: {model_key.upper()} ---")
        rss_before = get_process_rss_mb()
        
        # Instantiate model
        t0_load = time.perf_counter()
        model = model_cls()
        t_load = round(time.perf_counter() - t0_load, 3)
        rss_loaded = get_process_rss_mb()
        model_memory_mb = max(round(rss_loaded - rss_before, 2), 0.0)

        # Encode full Phase 10A corpus
        t0_index = time.perf_counter()
        doc_embeddings = model.encode_texts(chunk_texts, batch_size=32)
        indexing_time = round(time.perf_counter() - t0_index, 3)
        rss_indexed = get_process_rss_mb()

        # Sanity checks
        sanity = model.run_sanity_checks(doc_embeddings)
        assert sanity["sanity_passed"], f"Sanity check failed for {model_key}: {sanity}"

        # Save isolated index
        model_idx_dir = indexes_dir / model_key
        model_idx_dir.mkdir(parents=True, exist_ok=True)
        emb_file = model_idx_dir / "embeddings.npy"
        np.save(emb_file, doc_embeddings)
        index_size_mb = round(emb_file.stat().st_size / (1024 * 1024), 2)

        meta_file = model_idx_dir / "index_meta.json"
        meta_payload = {
            "model_key": model_key,
            "metadata": model.get_model_metadata(),
            "corpus_version": "clinical-kb-v1",
            "corpus_sha256": corpus_hash,
            "total_chunks": len(chunks),
            "dimension": model.dimension,
            "sanity_checks": sanity,
            "created_at": "2026-09-29T00:00:00Z"
        }
        with open(meta_file, "w", encoding="utf-8") as mf:
            json.dump(meta_payload, mf, indent=2)

        # Benchmark query latencies and retrieval accuracy
        query_latencies = []
        search_latencies = []
        total_latencies = []
        rankings: List[List[str]] = []

        for q in queries:
            t0_q = time.perf_counter()
            q_emb = model.encode_queries([q])[0]
            t_q = time.perf_counter() - t0_q

            t0_s = time.perf_counter()
            search_res = dense_search(q_emb, doc_embeddings, chunks, top_k=20)
            t_s = time.perf_counter() - t0_s

            query_latencies.append(t_q * 1000.0)
            search_latencies.append(t_s * 1000.0)
            total_latencies.append((t_q + t_s) * 1000.0)

            rankings.append([cid for _, cid in search_res])

        # Compute retrieval quality metrics
        rec1 = evaluator.compute_recall_at_k(rankings, target_hit_sets, k=1)
        rec5 = evaluator.compute_recall_at_k(rankings, target_hit_sets, k=5)
        rec10 = evaluator.compute_recall_at_k(rankings, target_hit_sets, k=10)
        rec20 = evaluator.compute_recall_at_k(rankings, target_hit_sets, k=20)
        mrr = evaluator.compute_mrr(rankings, target_first_hits)
        ndcg5 = evaluator.compute_ndcg(rankings, relevance_maps, k=5)
        ndcg10 = evaluator.compute_ndcg(rankings, relevance_maps, k=10)

        # Latency percentiles
        p50 = round(float(np.percentile(total_latencies, 50)), 2)
        p95 = round(float(np.percentile(total_latencies, 95)), 2)
        p99 = round(float(np.percentile(total_latencies, 99)), 2)

        all_model_results[model_key] = {
            "model_metadata": model.get_model_metadata(),
            "dense_metrics": {
                "recall_at_1": rec1,
                "recall_at_5": rec5,
                "recall_at_10": rec10,
                "recall_at_20": rec20,
                "mrr": mrr,
                "ndcg_at_5": ndcg5,
                "ndcg_at_10": ndcg10
            },
            "rankings": rankings,
            "sanity_checks": sanity
        }

        chunks_per_sec = round(len(chunks) / max(indexing_time, 0.001), 1)
        performance_records[model_key] = {
            "dimension": model.dimension,
            "indexing_time_sec": indexing_time,
            "chunks_per_sec": chunks_per_sec,
            "index_size_mb": index_size_mb,
            "model_load_time_sec": t_load,
            "peak_rss_mb": rss_indexed,
            "query_latency_ms": {
                "mean": round(float(np.mean(total_latencies)), 2),
                "p50": p50,
                "p95": p95,
                "p99": p99
            }
        }

        logger.info(
            f"[{model_key.upper()}] Recall@5={rec5:.4f}, MRR={mrr:.4f}, nDCG@5={ndcg5:.4f} | "
            f"IndexTime={indexing_time}s ({chunks_per_sec} ch/s) | Latency p50={p50}ms"
        )

    # Save output comparison files
    comparison_payload = {
        "benchmark_name": "Phase 10B Dense Medical Embedding Comparison",
        "corpus_version": "clinical-kb-v1",
        "corpus_sha256": corpus_hash,
        "eval_dataset_sha256": eval_hash,
        "query_count": len(queries),
        "total_corpus_chunks": len(chunks),
        "models": {k: v["dense_metrics"] for k, v in all_model_results.items()},
        "model_metadata": {k: v["model_metadata"] for k, v in all_model_results.items()},
        "sanity_checks": {k: v["sanity_checks"] for k, v in all_model_results.items()}
    }
    save_json(comparison_payload, output_dir / "phase10b_embedding_comparison.json")
    save_json(performance_records, output_dir / "phase10b_performance_results.json")

    return {
        "results": all_model_results,
        "performance": performance_records,
        "chunks": chunks,
        "eval_samples": eval_samples
    }


if __name__ == "__main__":
    p_root = Path(__file__).resolve().parent.parent.parent
    c_path = p_root / "data" / "processed" / "knowledge_base" / "clinical_chunks.json"
    e_path = p_root / "evaluation" / "datasets" / "rag_eval.jsonl"
    i_dir = p_root / "data" / "indexes"
    o_dir = p_root / "reports" / "evaluation"

    run_embedding_benchmark(c_path, e_path, i_dir, o_dir)
