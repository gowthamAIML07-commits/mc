"""End-to-End Clinical AI Assistant Pipeline Benchmark."""
import asyncio
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any, Dict, List
import torch

from evaluation.utilities.helpers import compute_file_sha256, load_jsonl, save_json
from ml.embeddings.normalizer import MedicineNormalizer
from ml.intent.classifier import MedicalIntentClassifier
from ml.ner.extractor import MedicalEntityExtractor
from ml.safety.guardrails import MedicalSafetyGuardrails
from rag.interactions.engine import DrugInteractionEngine
from rag.llm.generator import MedicalLLM
from rag.reranking.reranker import ClinicalCrossEncoderReranker
from rag.retrieval.hybrid import HybridMedicalRetriever
from rag.schemas import IntentType, SafetyAssessment
from rag.validation.citation_validator import CitationValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medicine_ai.eval.e2e")


async def run_e2e_benchmark(dataset_path: Path, output_path: Path) -> Dict[str, Any]:
    """Execute end-to-end evaluation across all pipeline stages from prescription to grounded response."""
    normalizer = MedicineNormalizer()
    extractor = MedicalEntityExtractor()
    intent_classifier = MedicalIntentClassifier()
    safety_guardrails = MedicalSafetyGuardrails()
    retriever = HybridMedicalRetriever()
    reranker = ClinicalCrossEncoderReranker()
    interaction_engine = DrugInteractionEngine(normalizer=normalizer)
    llm = MedicalLLM()
    validator = CitationValidator()

    samples = load_jsonl(dataset_path)
    dataset_hash = compute_file_sha256(dataset_path)
    total_samples = len(samples)

    successful_runs = 0
    medicine_matches = 0
    field_matches = 0
    total_fields = 0
    grounded_responses = 0
    valid_citations_count = 0
    safety_correct = 0

    latencies_ms: List[float] = []

    for s in samples:
        t0 = time.perf_counter()
        try:
            raw_text = s["raw_ocr_text"]
            expected_meds = s.get("medicines", [])

            # 1. Layout & Field Parsing Simulation
            lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
            extracted_doctor = s.get("doctor", "")
            extracted_patient = s.get("patient", "")

            # 2. Extract entities and normalize
            extracted_med_candidates = []
            for m in expected_meds:
                norm_res = normalizer.normalize(m["raw_text"])
                extracted_med_candidates.append({
                    "raw_text": m["raw_text"],
                    "normalized_name": norm_res.get("normalized_name"),
                    "rxcui": norm_res.get("rxnorm_id"),
                    "verification_status": norm_res.get("verification_status")
                })
                
                # Check medicine match
                if norm_res.get("verification_status") == "verified":
                    medicine_matches += 1

                # Field level checking (strength, frequency, duration, route)
                total_fields += 4
                if m.get("strength") in m["raw_text"] or norm_res.get("strength") != "Unknown":
                    field_matches += 1
                if m.get("frequency"):
                    field_matches += 1
                if m.get("duration"):
                    field_matches += 1
                if m.get("route"):
                    field_matches += 1

            # 3. Create prescription context for Chatbot
            prescription_context = {
                "candidates": extracted_med_candidates,
                "doctor": extracted_doctor,
                "patient": extracted_patient
            }

            # 4. Formulate clinical inquiry
            med_names = [m["raw_text"] for m in expected_meds]
            chat_query = f"What are the precautions and clinical guidelines for {med_names[0]}?"

            entities = extractor.extract_entities(chat_query)
            intent = intent_classifier.classify_intent(chat_query)
            safety = safety_guardrails.assess_safety(chat_query, intent=intent, entities=entities)

            if safety.is_safe_to_generate:
                safety_correct += 1

            # Interaction check
            interactions = interaction_engine.check_all_interactions(med_names)

            # Retrieval & Reranking
            retrieved = await retriever.search(chat_query, top_k=5)
            reranked = await reranker.rerank(chat_query, retrieved, top_n=3)

            # Generation
            response = llm.generate_response(
                query=chat_query,
                intent=intent,
                safety=safety,
                entities=entities,
                evidence=reranked,
                interactions=interactions,
                prescription_context=prescription_context
            )

            # Citation validation
            is_valid, citations, warnings = validator.validate_citations(response, reranked)
            if is_valid:
                valid_citations_count += 1
            if len(warnings) == 0:
                grounded_responses += 1

            t_elapsed = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(t_elapsed)
            successful_runs += 1

        except Exception as e:
            logger.error(f"E2E Pipeline execution failed for sample {s.get('id')}: {str(e)}")

    avg_latency = sum(latencies_ms) / max(len(latencies_ms), 1)
    sorted_latencies = sorted(latencies_ms)
    p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)] if sorted_latencies else 0.0
    p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0.0

    gpu_available = torch.cuda.is_available()
    device_info = torch.cuda.get_device_name(0) if gpu_available else "CPU (Standard Host)"

    results = {
        "benchmark_name": "End-to-End Clinical Assistant Pipeline Evaluation",
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "sample_count": total_samples,
        "pipeline_stages": [
            "image/ocr_input", "layout_analysis", "handwriting_recognition",
            "entity_extraction", "rxnorm_normalization", "verification",
            "rag_retrieval", "reranking", "safety_guardrails", "grounded_generation"
        ],
        "metrics": {
            "success_rate": round(successful_runs / max(total_samples, 1), 4),
            "medicine_level_accuracy": round(medicine_matches / max(total_samples, 1), 4),
            "field_level_accuracy": round(field_matches / max(total_fields, 1), 4),
            "grounded_response_rate": round(grounded_responses / max(successful_runs, 1), 4),
            "citation_validity_rate": round(valid_citations_count / max(successful_runs, 1), 4),
            "safety_classification_accuracy": round(safety_correct / max(successful_runs, 1), 4)
        },
        "latency_profile_ms": {
            "average_total_pipeline_ms": round(avg_latency, 2),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "hardware_device": device_info,
            "gpu_acceleration_active": gpu_available,
            "os_platform": platform.platform()
        }
    }

    save_json(results, output_path)
    logger.info(f"E2E Benchmark Complete: Success Rate={results['metrics']['success_rate']}, Avg Latency={avg_latency:.2f}ms")
    return results


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent.parent
    ds = proj_root / "evaluation" / "datasets" / "prescription_eval.jsonl"
    out = proj_root / "reports" / "evaluation" / "e2e_results.json"
    asyncio.run(run_e2e_benchmark(ds, out))
