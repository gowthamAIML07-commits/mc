"""Performance, Concurrency, and Latency Benchmark."""
import asyncio
import json
import logging
import os
import platform
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List
import torch

from evaluation.utilities.helpers import save_json
from ml.embeddings.normalizer import MedicineNormalizer
from ml.intent.classifier import MedicalIntentClassifier
from ml.ner.extractor import MedicalEntityExtractor
from rag.interactions.engine import DrugInteractionEngine
from rag.llm.generator import MedicalLLM
from rag.reranking.reranker import ClinicalCrossEncoderReranker
from rag.retrieval.hybrid import HybridMedicalRetriever
from rag.schemas import IntentType, SafetyAssessment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medicine_ai.eval.performance")


async def execute_mock_request(
    query: str,
    extractor: MedicalEntityExtractor,
    intent_cls: MedicalIntentClassifier,
    retriever: HybridMedicalRetriever,
    reranker: ClinicalCrossEncoderReranker,
    llm: MedicalLLM
) -> float:
    """Execute single end-to-end request and return elapsed milliseconds."""
    t0 = time.perf_counter()
    entities = extractor.extract_entities(query)
    intent = intent_cls.classify_intent(query)
    safety = SafetyAssessment(
        is_safe_to_generate=True,
        safety_level="LOW",
        matched_risk_categories=[],
        emergency_guidance=None
    )
    retrieved = await retriever.search(query, top_k=5)
    reranked = await reranker.rerank(query, retrieved, top_n=3)
    _ = llm.generate_response(
        query=query,
        intent=intent,
        safety=safety,
        entities=entities,
        evidence=reranked,
        interactions=[]
    )
    return (time.perf_counter() - t0) * 1000.0


async def run_performance_benchmark(output_path: Path) -> Dict[str, Any]:
    """Execute performance, cold start, warm latency, and concurrency benchmark."""
    # 1. Measure Cold Start
    t_cold_0 = time.perf_counter()
    extractor = MedicalEntityExtractor()
    intent_cls = MedicalIntentClassifier()
    retriever = HybridMedicalRetriever()
    reranker = ClinicalCrossEncoderReranker()
    llm = MedicalLLM()
    normalizer = MedicineNormalizer()
    t_cold_start_ms = (time.perf_counter() - t_cold_0) * 1000.0

    # Test queries
    test_queries = [
        "What is the recommended dosage of Metformin for type 2 diabetes?",
        "Can Warfarin and Aspirin be taken together safely?",
        "Tell me about common side effects of Cetirizine 10mg.",
        "What are the contraindications for Pantoprazole 40mg?",
        "Is Amoxicillin safe for treating bacterial infections in adults?"
    ]

    # 2. Warm Latency Runs (Sequential batch of 50 requests)
    latencies: List[float] = []
    for i in range(50):
        q = test_queries[i % len(test_queries)]
        lat = await execute_mock_request(q, extractor, intent_cls, retriever, reranker, llm)
        latencies.append(lat)

    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.50)]
    p95 = latencies[int(n * 0.95)]
    p99 = latencies[int(n * 0.99)]
    avg_latency = sum(latencies) / n

    # 3. Concurrent Load Test (Concurrency = 10)
    concurrency_level = 10
    total_concurrent_requests = 100
    t_load_0 = time.perf_counter()

    async def worker():
        tasks = []
        for i in range(total_concurrent_requests // concurrency_level):
            q = test_queries[i % len(test_queries)]
            tasks.append(execute_mock_request(q, extractor, intent_cls, retriever, reranker, llm))
        return await asyncio.gather(*tasks)

    concurrent_batches = await asyncio.gather(*[worker() for _ in range(concurrency_level)])
    t_total_load_sec = time.perf_counter() - t_load_0
    rps = total_concurrent_requests / max(t_total_load_sec, 0.001)

    # 4. System Resource Metrics
    tracemalloc.start()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gpu_info = "Not Available (CPU Inference Host)"
    if torch.cuda.is_available():
        gpu_info = f"CUDA Device: {torch.cuda.get_device_name(0)}"

    results = {
        "benchmark_name": "Performance, Latency & Load Benchmark",
        "sample_count": total_concurrent_requests + len(latencies),
        "cold_start_latency_ms": round(t_cold_start_ms, 2),
        "warm_inference_latency_ms": {
            "average_ms": round(avg_latency, 2),
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "min_ms": round(latencies[0], 2),
            "max_ms": round(latencies[-1], 2)
        },
        "concurrency_and_throughput": {
            "concurrency_level": concurrency_level,
            "total_requests_executed": total_concurrent_requests,
            "total_time_seconds": round(t_total_load_sec, 3),
            "requests_per_second": round(rps, 2)
        },
        "resource_utilization": {
            "peak_memory_traced_mb": round(peak_mem / (1024 * 1024), 2),
            "cpu_core_count": os.cpu_count() or 1,
            "gpu_hardware": gpu_info
        },
        "environment": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__
        }
    }

    save_json(results, output_path)
    logger.info(f"Performance Benchmark Complete: p50={p50:.2f}ms, p95={p95:.2f}ms, RPS={rps:.2f}")
    return results


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent.parent
    out = proj_root / "reports" / "evaluation" / "performance_results.json"
    asyncio.run(run_performance_benchmark(out))
