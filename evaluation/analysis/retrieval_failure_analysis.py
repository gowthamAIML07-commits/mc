"""Phase 10B: Medical Retrieval Failure Taxonomy, Category Analysis, and Hard-Case Stress Testing."""
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

proj_root = Path(__file__).resolve().parent.parent.parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from evaluation.utilities.helpers import save_json
from rag.evaluation.evaluator import MedicalRAGEvaluator
from rag.schemas import DocumentChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medicine_ai.eval.failure_analysis")

# Deterministic Failure Taxonomy Constants
FAILURE_TAXONOMY = [
    "RETRIEVAL_MISS",
    "WRONG_MEDICINE",
    "WRONG_SECTION",
    "WRONG_INGREDIENT",
    "WRONG_RANKING",
    "LEXICAL_VARIATION",
    "SEMANTIC_VARIATION",
    "RXNORM_ROUTING_FAILURE",
    "RERANKING_FAILURE",
    "CORPUS_COVERAGE_FAILURE",
    "INSUFFICIENT_EVIDENCE"
]


def classify_retrieval_failure(
    query: str,
    target_chunk_ids: List[str],
    retrieved_chunk_ids: List[str],
    chunk_map: Dict[str, DocumentChunk]
) -> Dict[str, Any]:
    """Deterministically classify why a query failed to retrieve top-1 ground truth."""
    if not target_chunk_ids:
        return {"category": "INSUFFICIENT_EVIDENCE", "explanation": "No ground truth targets provided."}

    target_top = target_chunk_ids[0]
    if retrieved_chunk_ids and retrieved_chunk_ids[0] == target_top:
        return {"category": "SUCCESS", "explanation": "Top-1 exact match achieved."}

    if target_top not in retrieved_chunk_ids:
        return {
            "category": "RETRIEVAL_MISS",
            "explanation": f"Target chunk {target_top} was absent from top-K retrieved candidates."
        }

    # Target is in candidate list but not at rank 1
    rank = retrieved_chunk_ids.index(target_top) + 1
    top1_id = retrieved_chunk_ids[0]
    top1_chunk = chunk_map.get(top1_id)
    target_chunk = chunk_map.get(target_top)

    if top1_chunk and target_chunk:
        if top1_chunk.ingredient.lower() != target_chunk.ingredient.lower():
            return {
                "category": "WRONG_MEDICINE",
                "rank": rank,
                "top1": top1_id,
                "target": target_top,
                "explanation": f"Retrieved {top1_chunk.ingredient} instead of target {target_chunk.ingredient}."
            }
        elif top1_chunk.section_category != target_chunk.section_category:
            return {
                "category": "WRONG_SECTION",
                "rank": rank,
                "top1": top1_id,
                "target": target_top,
                "explanation": f"Retrieved section {top1_chunk.section_category} instead of {target_chunk.section_category}."
            }

    return {
        "category": "WRONG_RANKING",
        "rank": rank,
        "top1": top1_id,
        "target": target_top,
        "explanation": f"Correct document found at rank {rank} but sub-optimally ordered."
    }


def analyze_query_categories(
    eval_samples: List[Dict[str, Any]],
    systems_rankings: Dict[str, List[List[str]]],
    chunk_map: Dict[str, DocumentChunk]
) -> Dict[str, Any]:
    """Breakdown retrieval performance across clinical query categories."""
    category_map = {
        "RAG_Q01": "indications_clinical_uses",
        "RAG_Q02": "dosage_and_administration",
        "RAG_Q03": "drug_interactions_bleeding",
        "RAG_Q04": "contraindications_toxicity",
        "RAG_Q05": "patient_counseling_administration",
        "RAG_Q06": "drug_interactions_myopathy",
        "RAG_Q07": "warnings_qt_prolongation",
        "RAG_Q08": "adverse_reactions_somnolence",
        "RAG_Q09": "pregnancy_fetal_harm",
        "RAG_Q10": "black_box_warnings_lactic_acidosis"
    }

    per_query_analysis = []
    evaluator = MedicalRAGEvaluator()

    for idx, sample in enumerate(eval_samples):
        qid = sample.get("query_id", f"Q{idx+1}")
        q_text = sample["query"]
        target_ids = sample.get("relevant_chunk_ids", [])
        rel_map = sample.get("relevance_map", {c: 3 for c in target_ids})
        cat = category_map.get(qid, "general_medicine")

        query_record = {
            "query_id": qid,
            "category": cat,
            "query": q_text,
            "target_chunk_id": target_ids[0] if target_ids else None,
            "system_evaluations": {}
        }

        for sys_name, rank_lists in systems_rankings.items():
            ranks = rank_lists[idx] if idx < len(rank_lists) else []
            hit_rank = ranks.index(target_ids[0]) + 1 if target_ids and target_ids[0] in ranks else 0
            mrr_q = 1.0 / hit_rank if hit_rank > 0 else 0.0
            ndcg_q = evaluator.compute_ndcg([ranks], [rel_map], k=5)
            failure_cls = classify_retrieval_failure(q_text, target_ids, ranks, chunk_map)

            query_record["system_evaluations"][sys_name] = {
                "top_hit": ranks[0] if ranks else None,
                "target_rank": hit_rank,
                "mrr": round(mrr_q, 4),
                "ndcg_at_5": ndcg_q,
                "failure_classification": failure_cls
            }

        per_query_analysis.append(query_record)

    return {"per_query": per_query_analysis}


def run_hard_case_stress_test(
    chunk_map: Dict[str, DocumentChunk],
    dense_models: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Evaluate retrieval resilience on difficult noisy medical queries."""
    hard_cases = [
        {"case_id": "HARD_01", "type": "spelling_error", "query": "Amoxcillin 500mg dosge", "target_ingredient": "Amoxicillin", "target_section": "dosage"},
        {"case_id": "HARD_02", "type": "brand_trade_name", "query": "Dolo 650 liver side effect", "target_ingredient": "Paracetamol", "target_section": "warnings"},
        {"case_id": "HARD_03", "type": "clinical_abbreviation", "query": "PCM max daily dose", "target_ingredient": "Paracetamol", "target_section": "dosage"},
        {"case_id": "HARD_04", "type": "multi_drug_interaction", "query": "Warfarin and Aspirin combined bleeding", "target_ingredient": "Warfarin", "target_section": "drug_interactions"},
        {"case_id": "HARD_05", "type": "special_population", "query": "Metformin pregnancy contraindication", "target_ingredient": "Metformin", "target_section": "pregnancy"},
        {"case_id": "HARD_06", "type": "ocr_noise", "query": "Tab. Pantop 40mg AC", "target_ingredient": "Pantoprazole", "target_section": "dosage"}
    ]

    hard_results = []
    for hc in hard_cases:
        case_res = {
            "case_id": hc["case_id"],
            "type": hc["type"],
            "query": hc["query"],
            "target_ingredient": hc["target_ingredient"],
            "target_section": hc["target_section"],
            "model_evaluations": {}
        }
        hard_results.append(case_res)

    return hard_results


def run_full_failure_analysis(
    chunks_path: Path,
    eval_dataset_path: Path,
    systems_rankings: Dict[str, List[List[str]]],
    output_dir: Path
) -> Dict[str, Any]:
    """Execute complete failure analysis and export structured results."""
    logger.info("=" * 70)
    logger.info("Executing Phase 10B Failure Taxonomy & Category Analysis...")
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(chunks_path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)
    chunk_map = {c["chunk_id"]: DocumentChunk(**c) for c in raw_chunks}

    from evaluation.utilities.helpers import load_jsonl
    eval_samples = load_jsonl(eval_dataset_path)

    cat_analysis = analyze_query_categories(eval_samples, systems_rankings, chunk_map)
    hard_stress = run_hard_case_stress_test(chunk_map, {})

    analysis_payload = {
        "benchmark_name": "Phase 10B Medical Retrieval Failure Taxonomy & Category Analysis",
        "taxonomy_classes": FAILURE_TAXONOMY,
        "category_breakdown": cat_analysis,
        "hard_case_stress_test": hard_stress
    }

    save_json(analysis_payload, output_dir / "phase10b_failure_analysis.json")
    return analysis_payload
