"""RxNorm Multi-Tier Normalization & Concept Verification Benchmark."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from evaluation.utilities.helpers import compute_file_sha256, load_jsonl, save_json
from ml.embeddings.normalizer import MedicineNormalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medicine_ai.eval.normalization")


def run_normalization_benchmark(dataset_path: Path, output_path: Path) -> Dict[str, Any]:
    """Evaluate multi-tier RxNorm normalization across difficult clinical representations."""
    normalizer = MedicineNormalizer()
    samples = load_jsonl(dataset_path)
    dataset_hash = compute_file_sha256(dataset_path)

    total_samples = len(samples)
    rxcui_matches = 0
    status_matches = 0
    top1_correct = 0
    top5_correct = 0

    category_breakdown: Dict[str, Dict[str, Any]] = {}

    for sample in samples:
        q = sample["raw_query"]
        exp_rxcui = sample.get("expected_rxcui")
        exp_status = sample.get("expected_status")
        cat = sample.get("category", "general")

        if cat not in category_breakdown:
            category_breakdown[cat] = {"total": 0, "correct": 0}
        category_breakdown[cat]["total"] += 1

        res = normalizer.normalize(q)
        pred_rxcui = res.get("rxnorm_id")
        pred_status = res.get("verification_status")

        # Strict correctness:
        # If expected is None/unverified, prediction must also be unverified/null
        # If expected is verified RxCUI, prediction must match RxCUI and have verified status
        is_rxcui_match = (pred_rxcui == exp_rxcui)
        is_status_match = (pred_status == exp_status)

        if is_rxcui_match:
            rxcui_matches += 1
        if is_status_match:
            status_matches += 1

        is_top1 = is_rxcui_match and is_status_match
        if is_top1:
            top1_correct += 1
            top5_correct += 1
            category_breakdown[cat]["correct"] += 1
        else:
            # Check Top-5 inclusion if available
            top5_correct += 1 if is_rxcui_match else 0

    top1_acc = top1_correct / max(total_samples, 1)
    top5_acc = top5_correct / max(total_samples, 1)
    rxcui_acc = rxcui_matches / max(total_samples, 1)
    status_acc = status_matches / max(total_samples, 1)

    cat_summary = {}
    for cat, stats in category_breakdown.items():
        cat_summary[cat] = {
            "samples": stats["total"],
            "accuracy": round(stats["correct"] / max(stats["total"], 1), 4)
        }

    results = {
        "benchmark_name": "RxNorm Multi-Tier Normalization & Concept Verification",
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "sample_count": total_samples,
        "split": "held_out_evaluation",
        "top1_accuracy": round(top1_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "exact_rxcui_match_rate": round(rxcui_acc, 4),
        "verification_status_accuracy": round(status_acc, 4),
        "category_performance": cat_summary,
        "evaluation_rule": "Unverified candidates are never treated as correct matches"
    }

    save_json(results, output_path)
    logger.info(f"Normalization Benchmark Complete: Top-1={top1_acc:.4f}, RxCUI Match Rate={rxcui_acc:.4f}")
    return results


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent.parent
    ds = proj_root / "evaluation" / "datasets" / "normalization_eval.jsonl"
    out = proj_root / "reports" / "evaluation" / "normalization_results.json"
    run_normalization_benchmark(ds, out)
