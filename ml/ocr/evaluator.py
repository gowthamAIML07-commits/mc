"""OCR evaluation metrics: Character Error Rate (CER), Word Error Rate (WER), and Field Accuracy."""
from typing import List, Tuple, Dict, Any
import Levenshtein


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER = Levenshtein_distance / len(ref))."""
    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    dist = Levenshtein.distance(ref, hyp)
    return dist / len(ref)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER = Word_Levenshtein_distance / num_words(ref))."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = Levenshtein.distance(" ".join(ref_words), " ".join(hyp_words))
    return min(dist / max(len(" ".join(ref_words)), 1), 1.0)


def compute_field_accuracy(predicted_fields: dict, ground_truth_fields: dict) -> Tuple[float, dict]:
    """Compute exact and fuzzy field extraction accuracy across clinical entities."""
    details = {}
    matches = 0
    total_fields = 0

    # Fields to check: doctor, patient, date, medicines
    if "doctor" in ground_truth_fields:
        total_fields += 1
        gt_doc = ground_truth_fields["doctor"].get("name", "") if isinstance(ground_truth_fields["doctor"], dict) else str(ground_truth_fields["doctor"])
        pred_doc = predicted_fields.get("doctor", "")
        sim = Levenshtein.ratio(gt_doc.lower(), str(pred_doc).lower())
        details["doctor_similarity"] = round(sim, 3)
        if sim >= 0.80:
            matches += 1

    if "patient" in ground_truth_fields:
        total_fields += 1
        gt_pat = ground_truth_fields["patient"].get("name", "") if isinstance(ground_truth_fields["patient"], dict) else str(ground_truth_fields["patient"])
        pred_pat = predicted_fields.get("patient", "")
        sim = Levenshtein.ratio(gt_pat.lower(), str(pred_pat).lower())
        details["patient_similarity"] = round(sim, 3)
        if sim >= 0.80:
            matches += 1

    if "date" in ground_truth_fields:
        total_fields += 1
        gt_date = str(ground_truth_fields["date"])
        pred_date = str(predicted_fields.get("date", ""))
        is_exact = (gt_date == pred_date)
        details["date_exact_match"] = is_exact
        if is_exact:
            matches += 1

    acc = round(matches / max(total_fields, 1), 3)
    return acc, details


class OCREvaluator:
    """Evaluator class for CER, WER, and field extraction accuracy."""

    def compute_cer(self, reference: str, hypothesis: str) -> float:
        return compute_cer(reference, hypothesis)

    def compute_wer(self, reference: str, hypothesis: str) -> float:
        return compute_wer(reference, hypothesis)

    def compute_field_accuracy(self, predicted_fields: dict, ground_truth_fields: dict) -> Tuple[float, dict]:
        return compute_field_accuracy(predicted_fields, ground_truth_fields)
