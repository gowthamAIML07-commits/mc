"""Clinical Safety Guardrails & Emergency Detection Benchmark."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from evaluation.utilities.helpers import compute_file_sha256, load_jsonl, save_json
from ml.safety.guardrails import MedicalSafetyGuardrails

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medicine_ai.eval.safety")


def run_safety_benchmark(dataset_path: Path, output_path: Path) -> Dict[str, Any]:
    """Evaluate safety classifier, emergency detection sensitivity, and risk category routing."""
    guardrails = MedicalSafetyGuardrails()
    samples = load_jsonl(dataset_path)
    dataset_hash = compute_file_sha256(dataset_path)

    total_samples = len(samples)
    tp, fp, fn, tn = 0, 0, 0, 0
    correct_levels = 0

    emergency_false_negatives: List[Dict[str, Any]] = []
    level_confusion: Dict[str, Dict[str, int]] = {}

    for s in samples:
        text = s["text"]
        exp_level = s["expected_safety_level"]
        exp_emergency = s.get("is_emergency", False)
        exp_intent = s.get("expected_intent")

        assessment = guardrails.assess_safety(text, intent=exp_intent)
        pred_level = assessment.safety_level
        pred_emergency = assessment.is_emergency

        # Confusion matrix by safety level
        if exp_level not in level_confusion:
            level_confusion[exp_level] = {}
        level_confusion[exp_level][pred_level] = level_confusion[exp_level].get(pred_level, 0) + 1

        if pred_level == exp_level:
            correct_levels += 1

        # Emergency binary detection metrics
        if pred_emergency and exp_emergency:
            tp += 1
        elif pred_emergency and not exp_emergency:
            fp += 1
        elif not pred_emergency and exp_emergency:
            fn += 1
            emergency_false_negatives.append({
                "test_id": s.get("id"),
                "text": text,
                "predicted_level": pred_level,
                "expected_level": exp_level
            })
        else:
            tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
    fnr = fn / max(fn + tp, 1)

    results = {
        "benchmark_name": "Clinical Safety Guardrails & Emergency Detection Benchmark",
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "sample_count": total_samples,
        "emergency_detection_metrics": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_negative_rate": round(fnr, 4)
        },
        "overall_safety_level_accuracy": round(correct_levels / max(total_samples, 1), 4),
        "emergency_false_negatives_count": len(emergency_false_negatives),
        "emergency_false_negatives_details": emergency_false_negatives,
        "safety_level_confusion_matrix": level_confusion,
        "clinical_validation_disclaimer": "CRITICAL: This benchmark evaluates deterministic safety guardrail rules on curated test cases. It does NOT constitute clinical trial validation or comprehensive real-world hospital deployment certification."
    }

    save_json(results, output_path)
    logger.info(f"Safety Benchmark Complete: Emergency Recall={recall:.4f}, False Negative Rate={fnr:.4f}")
    return results


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent.parent
    ds = proj_root / "evaluation" / "datasets" / "safety_eval.jsonl"
    out = proj_root / "reports" / "evaluation" / "safety_results.json"
    run_safety_benchmark(ds, out)
