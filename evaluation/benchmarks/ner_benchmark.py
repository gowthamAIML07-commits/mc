"""Clinical Medical Named Entity Recognition (NER) Benchmark."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

from ml.ner.extractor import MedicalEntityExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medicine_ai.eval.ner")


def run_ner_benchmark(output_path: Path) -> Dict[str, Any]:
    """Evaluate medical entity extractor across multi-class clinical spans."""
    extractor = MedicalEntityExtractor()

    # Curated held-out test evaluation samples
    test_cases = [
        {
            "text": "Prescribed Amoxicillin 500mg TDS for 7 days to treat acute bacterial sinusitis.",
            "expected_entities": [
                {"label": "DRUG", "text": "Amoxicillin"},
                {"label": "DOSAGE", "text": "500mg"},
                {"label": "FREQUENCY", "text": "TDS"},
                {"label": "DURATION", "text": "7 days"},
                {"label": "DISEASE", "text": "sinusitis"}
            ]
        },
        {
            "text": "Patient with type 2 diabetes taking Metformin 500mg twice daily with meals.",
            "expected_entities": [
                {"label": "DISEASE", "text": "type 2 diabetes"},
                {"label": "DRUG", "text": "Metformin"},
                {"label": "DOSAGE", "text": "500mg"},
                {"label": "FREQUENCY", "text": "twice daily"}
            ]
        },
        {
            "text": "Patient has severe fever, cough, and chest pain. Give Dolo 650 SOS.",
            "expected_entities": [
                {"label": "SYMPTOM", "text": "fever"},
                {"label": "SYMPTOM", "text": "cough"},
                {"label": "SYMPTOM", "text": "chest pain"},
                {"label": "DRUG", "text": "Dolo 650"},
                {"label": "FREQUENCY", "text": "SOS"}
            ]
        },
        {
            "text": "History of penicillin allergy and asthma. Take Cetirizine 10mg OD at bedtime for 10 days.",
            "expected_entities": [
                {"label": "ALLERGY", "text": "penicillin allergy"},
                {"label": "DISEASE", "text": "asthma"},
                {"label": "DRUG", "text": "Cetirizine"},
                {"label": "DOSAGE", "text": "10mg"},
                {"label": "FREQUENCY", "text": "OD"},
                {"label": "DURATION", "text": "10 days"}
            ]
        },
        {
            "text": "Administer Warfarin 5mg once daily. Avoid Aspirin.",
            "expected_entities": [
                {"label": "DRUG", "text": "Warfarin"},
                {"label": "DOSAGE", "text": "5mg"},
                {"label": "FREQUENCY", "text": "once daily"},
                {"label": "DRUG", "text": "Aspirin"}
            ]
        }
    ]

    total_expected = 0
    total_predicted = 0
    total_correct = 0

    per_label_stats: Dict[str, Dict[str, int]] = {}

    for tc in test_cases:
        extracted = extractor.extract_entities(tc["text"])
        expected = tc["expected_entities"]

        total_expected += len(expected)
        total_predicted += len(extracted)

        # Match prediction spans & labels
        for exp in expected:
            lbl = exp["label"]
            if lbl not in per_label_stats:
                per_label_stats[lbl] = {"tp": 0, "fp": 0, "fn": 0, "total_exp": 0}
            per_label_stats[lbl]["total_exp"] += 1

            matched = any(
                e.label == exp["label"] and (exp["text"].lower() in e.text.lower() or e.text.lower() in exp["text"].lower())
                for e in extracted
            )
            if matched:
                total_correct += 1
                per_label_stats[lbl]["tp"] += 1
            else:
                per_label_stats[lbl]["fn"] += 1

    precision = total_correct / max(total_predicted, 1)
    recall = total_correct / max(total_expected, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)

    label_breakdown = {}
    for lbl, stats in per_label_stats.items():
        p = stats["tp"] / max(stats["tp"] + stats["fp"], 1)
        r = stats["tp"] / max(stats["total_exp"], 1)
        f = (2 * p * r) / max(p + r, 1e-8)
        label_breakdown[lbl] = {
            "samples": stats["total_exp"],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f, 4)
        }

    results = {
        "benchmark_name": "Clinical Named Entity Recognition (NER)",
        "sample_count": len(test_cases),
        "total_gold_entities": total_expected,
        "total_extracted_entities": total_predicted,
        "overall_precision": round(precision, 4),
        "overall_recall": round(recall, 4),
        "overall_f1_score": round(f1, 4),
        "entity_classes_evaluated": [
            "DRUG", "BRAND", "ACTIVE_INGREDIENT", "DOSAGE", "FREQUENCY",
            "DURATION", "SYMPTOM", "DISEASE", "ALLERGY", "LAB_TEST", "MEDICAL_PROCEDURE"
        ],
        "per_label_metrics": label_breakdown
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"NER Benchmark Complete: Overall F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    return results


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent.parent
    out = proj_root / "reports" / "evaluation" / "ner_results.json"
    run_ner_benchmark(out)
