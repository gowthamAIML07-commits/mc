"""Unit tests for individual training pipelines and ML model components."""
import pytest
import torch
from ml.ocr.evaluator import compute_cer, compute_wer, compute_field_accuracy
from ml.ocr.augmentation import SyntheticPrescriptionGenerator
from ml.ocr.layout_parser import SpatialRelationExtractor, extract_spatial_features
from ml.handwriting.model import CRNNHandwritingClassifier
from ml.embeddings.normalizer import MedicineNormalizer, simple_metaphone


def test_ocr_metrics_cer_and_wer():
    """Verify CER and WER computation correctness."""
    ref = "Amoxicillin 500mg"
    hyp_exact = "Amoxicillin 500mg"
    hyp_err = "Amoxcillin 500mg"

    assert compute_cer(ref, hyp_exact) == 0.0
    assert compute_wer(ref, hyp_exact) == 0.0
    assert 0.0 < compute_cer(ref, hyp_err) < 0.2


def test_field_accuracy_evaluation():
    """Verify structured field extraction accuracy evaluation."""
    gt = {
        "doctor": {"name": "Dr. R. K. Mukherjee"},
        "patient": {"name": "Aarav Sharma"},
        "date": "15/03/2024"
    }
    pred = {
        "doctor": "Dr. R. K. Mukherjee",
        "patient": "Aarav Sharma",
        "date": "15/03/2024"
    }
    acc, details = compute_field_accuracy(pred, gt)
    assert acc == 1.0
    assert details["doctor_similarity"] == 1.0
    assert details["date_exact_match"] is True


def test_synthetic_prescription_generator(tmp_path):
    """Verify synthetic generator generates valid images and annotations."""
    gen = SyntheticPrescriptionGenerator(output_dir=tmp_path, width=400, height=500)
    img, ann = gen.generate_single_prescription(index=0, seed=42)

    assert img.size == (400, 500)
    assert ann["prescription_id"] == "synth_rx_0000"
    assert "doctor" in ann
    assert "patient" in ann
    assert len(ann["medicines"]) >= 2


def test_crnn_model_forward_pass():
    """Verify CRNN architecture produces expected output tensor shapes."""
    batch_size = 4
    num_classes = 78
    model = CRNNHandwritingClassifier(num_classes=num_classes)
    dummy_input = torch.randn(batch_size, 3, 64, 192)
    logits = model(dummy_input)

    assert logits.shape == (batch_size, num_classes)


def test_spatial_relation_feature_extraction():
    """Verify 12-dimensional spatial geometric feature extraction."""
    b1 = [40, 50, 100, 70]
    b2 = [110, 50, 200, 70]
    feat = extract_spatial_features(b1, b2, page_w=1000, page_h=1000)

    assert feat.shape == (12,)
    assert feat.dtype == torch.float32

    model = SpatialRelationExtractor(num_categories=6)
    out = model(feat.unsqueeze(0))
    assert out.shape == (1, 6)


def test_medicine_normalizer_matching():
    """Verify 4-tier normalizer behavior across exact, phonetic, and fuzzy matches."""
    normalizer = MedicineNormalizer()
    assert simple_metaphone("Amoxicillin") == simple_metaphone("Amoxcillin")
    assert simple_metaphone("Cetirizine") == simple_metaphone("Cetrizine")
    
    # Test 4-tier normalization
    res_amox = normalizer.normalize("Amoxcillin 500mg")
    assert res_amox["rxnorm_id"] == "308189"
    assert res_amox["ingredient"] == "Amoxicillin"

    res_para = normalizer.normalize("Paractaml 650")
    assert res_para["rxnorm_id"] == "161"
    assert res_para["ingredient"] == "Paracetamol"
