"""Comprehensive deterministic and statistical auditor for MedOCR-Vision dataset."""
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import numpy as np
from PIL import Image
from datasets import load_from_disk
import Levenshtein


# Curated medical & prescription keyword lexicons for deterministic heuristics
PRESCRIPTION_KEYWORDS = {
    "rx", "tab", "cap", "tablet", "capsule", "syrup", "injection", "inj", "mg", "ml", "gm",
    "dr.", "dr", "doctor", "clinic", "hospital", "patient", "age", "gender", "date", "reg",
    "od", "bd", "tds", "qid", "sos", "hs", "po", "oral", "dose", "dosage", "duration", "days",
    "after food", "before food", "bedtime", "meals", "advice", "signature"
}

DRUG_KEYWORDS = {
    "amoxicillin", "paracetamol", "ibuprofen", "azithromycin", "pantoprazole", "metformin",
    "cetirizine", "omeprazole", "atorvastatin", "amlodipine", "ciprofloxacin", "losartan",
    "doxycycline", "prednisolone", "clavulanate", "augmentin", "dolo", "pan", "calpol"
}

CLINICAL_DOC_KEYWORDS = {
    "diagnosis", "symptoms", "history", "examination", "lab", "test", "report", "blood",
    "hemoglobin", "wbc", "rbc", "platelet", "urine", "ecg", "x-ray", "ct scan", "mri",
    "treatment", "discharge", "summary", "admission", "department", "physician", "pathology"
}


def hash_image(img: Image.Image) -> str:
    """Compute MD5 hash of raw image pixel bytes."""
    return hashlib.md5(img.tobytes()).hexdigest()


def hash_text(text: str) -> str:
    """Compute normalized text hash."""
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def classify_text_heuristically(text: str) -> str:
    """Deterministic heuristic classification based on keyword density and layout indicators."""
    t_lower = text.lower()
    words = set(re.findall(r"\b[a-z0-9\.]+\b", t_lower))

    rx_matches = len(words.intersection(PRESCRIPTION_KEYWORDS))
    drug_matches = len(words.intersection(DRUG_KEYWORDS))
    doc_matches = len(words.intersection(CLINICAL_DOC_KEYWORDS))

    # Check for Rx specific patterns (e.g., dosage numbers + frequencies)
    has_rx_symbol = bool(re.search(r"\b(rx|r/x|tab\.|cap\.|syr\.)\b", t_lower))
    has_dosage_pattern = bool(re.search(r"\b\d+\s*(mg|ml|gm)\b", t_lower))
    has_freq_pattern = bool(re.search(r"\b(1-0-1|1-1-1|1-0-0|0-0-1|o\.d|b\.d|t\.d\.s|s\.o\.s)\b", t_lower))

    if (has_rx_symbol or has_dosage_pattern or has_freq_pattern) and rx_matches >= 3:
        return "prescription"
    elif drug_matches >= 1 or has_dosage_pattern:
        return "medicine_drug_related"
    elif doc_matches >= 3 or rx_matches >= 2:
        return "medical_document"
    elif len(words) > 0:
        return "general_ocr"
    else:
        return "unknown"


def detect_script(text: str) -> str:
    """Detect dominant script in text."""
    if not text.strip():
        return "empty"
    latin_chars = len(re.findall(r"[a-zA-Z]", text))
    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
    bengali_chars = len(re.findall(r"[\u0980-\u09FF]", text))
    digits = len(re.findall(r"[0-9]", text))

    total = max(len(text), 1)
    if latin_chars / total > 0.4:
        return "latin"
    elif devanagari_chars / total > 0.3:
        return "devanagari"
    elif bengali_chars / total > 0.3:
        return "bengali"
    elif digits / total > 0.5:
        return "numeric_only"
    return "mixed_or_symbols"


def audit_medocr_dataset() -> Tuple[Dict[str, Any], str]:
    """Perform full audit of MedOCR-Vision dataset across all splits."""
    start_time = time.time()
    root_dir = Path(__file__).resolve().parent.parent
    data_path = root_dir / "data" / "raw" / "indian_medical_prescription_ocr"

    print(f"[*] Loading dataset from {data_path}...")
    ds = load_from_disk(str(data_path))

    audit_data = {
        "dataset_name": "MedOCR-Vision Dataset (Indian Medical Prescription OCR)",
        "source_url": "https://huggingface.co/datasets/naazimsnh02/medocr-vision-dataset",
        "verified_hash": "664201c9bb855547e626d8edcad4101da0eef113762606400284330279ef5e7b",
        "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_samples": sum(len(ds[s]) for s in ds.keys()),
        "splits": {}
    }

    # Tracking sets for cross-split leakage
    split_image_hashes: Dict[str, Set[str]] = {}
    split_text_hashes: Dict[str, Set[str]] = {}
    representative_samples: Dict[str, List[dict]] = {}

    for split_name in ["train", "validation", "test"]:
        split_ds = ds[split_name]
        count = len(split_ds)
        print(f"[*] Auditing split '{split_name}' ({count} samples)...")

        widths, heights, aspect_ratios = [], [], []
        image_modes, image_formats = Counter(), Counter()
        missing_images = 0
        corrupt_images = 0

        text_lengths = []
        word_counts = []
        line_counts = []
        empty_text_count = 0
        short_text_count = 0 # < 5 chars
        long_text_count = 0  # > 500 chars

        img_hashes = []
        txt_hashes = []
        categories = Counter()
        scripts = Counter()

        samples_inspected = []

        for idx, sample in enumerate(split_ds):
            img = sample.get("image")
            text = sample.get("text") or ""

            # 1. Image Inspection
            if img is None:
                missing_images += 1
                continue

            try:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(round(w / max(h, 1), 3))
                image_modes[img.mode] += 1
                image_formats[img.format or "PIL_IN_MEMORY"] += 1
                img_hash = hash_image(img)
                img_hashes.append(img_hash)
            except Exception:
                corrupt_images += 1
                continue

            # 2. Text Inspection
            text_str = str(text).strip()
            txt_len = len(text_str)
            text_lengths.append(txt_len)
            words = text_str.split()
            word_counts.append(len(words))
            line_counts.append(len(text_str.splitlines()))

            if not text_str:
                empty_text_count += 1
            elif txt_len < 5:
                short_text_count += 1
            elif txt_len > 500:
                long_text_count += 1

            txt_hash = hash_text(text_str)
            txt_hashes.append(txt_hash)

            # Heuristic classification & script
            cat = classify_text_heuristically(text_str)
            categories[cat] += 1

            scr = detect_script(text_str)
            scripts[scr] += 1

            # Store first 3 representative samples
            if idx < 3:
                samples_inspected.append({
                    "sample_index": idx,
                    "image_dimensions": [w, h],
                    "image_mode": img.mode,
                    "text_length": txt_len,
                    "text_snippet": text_str[:160] + ("..." if txt_len > 160 else ""),
                    "category": cat,
                    "script": scr
                })

        split_image_hashes[split_name] = set(img_hashes)
        split_text_hashes[split_name] = set(txt_hashes)
        representative_samples[split_name] = samples_inspected

        # Duplicate counts within split
        unique_imgs = len(set(img_hashes))
        unique_txts = len(set(txt_hashes))
        img_duplicates_within = count - unique_imgs
        txt_duplicates_within = count - unique_txts

        audit_data["splits"][split_name] = {
            "sample_count": count,
            "dimensions": {
                "min_width": int(np.min(widths)) if widths else 0,
                "max_width": int(np.max(widths)) if widths else 0,
                "mean_width": float(round(np.mean(widths), 1)) if widths else 0,
                "median_width": float(np.median(widths)) if widths else 0,
                "min_height": int(np.min(heights)) if heights else 0,
                "max_height": int(np.max(heights)) if heights else 0,
                "mean_height": float(round(np.mean(heights), 1)) if heights else 0,
                "median_height": float(np.median(heights)) if heights else 0,
                "mean_aspect_ratio": float(round(np.mean(aspect_ratios), 3)) if aspect_ratios else 0
            },
            "image_modes": dict(image_modes),
            "missing_images": missing_images,
            "corrupt_images": corrupt_images,
            "text_statistics": {
                "empty_text_count": empty_text_count,
                "short_text_count_lt_5": short_text_count,
                "long_text_count_gt_500": long_text_count,
                "min_chars": int(np.min(text_lengths)) if text_lengths else 0,
                "max_chars": int(np.max(text_lengths)) if text_lengths else 0,
                "mean_chars": float(round(np.mean(text_lengths), 1)) if text_lengths else 0,
                "median_chars": float(np.median(text_lengths)) if text_lengths else 0,
                "mean_words": float(round(np.mean(word_counts), 1)) if word_counts else 0,
                "mean_lines": float(round(np.mean(line_counts), 1)) if line_counts else 0
            },
            "duplicates_within_split": {
                "image_exact_duplicates": img_duplicates_within,
                "text_exact_duplicates": txt_duplicates_within
            },
            "category_distribution": dict(categories),
            "script_distribution": dict(scripts),
            "representative_samples": samples_inspected
        }

    # 3. Cross-Split Leakage Analysis
    train_imgs = split_image_hashes.get("train", set())
    val_imgs = split_image_hashes.get("validation", set())
    test_imgs = split_image_hashes.get("test", set())

    train_txts = split_text_hashes.get("train", set())
    val_txts = split_text_hashes.get("validation", set())
    test_txts = split_text_hashes.get("test", set())

    img_leak_train_val = len(train_imgs.intersection(val_imgs))
    img_leak_train_test = len(train_imgs.intersection(test_imgs))
    img_leak_val_test = len(val_imgs.intersection(test_imgs))

    txt_leak_train_val = len(train_txts.intersection(val_txts))
    txt_leak_train_test = len(train_txts.intersection(test_txts))
    txt_leak_val_test = len(val_txts.intersection(test_txts))

    audit_data["cross_split_leakage"] = {
        "exact_image_overlap": {
            "train_vs_validation": img_leak_train_val,
            "train_vs_test": img_leak_train_test,
            "validation_vs_test": img_leak_val_test
        },
        "exact_text_overlap": {
            "train_vs_validation": txt_leak_train_val,
            "train_vs_test": txt_leak_train_test,
            "validation_vs_test": txt_leak_val_test
        },
        "leakage_summary": "No exact image duplicates detected across splits." if (img_leak_train_val == 0 and img_leak_train_test == 0) else "Exact image overlap detected across splits."
    }

    # Overall Category Totals
    total_categories = Counter()
    total_scripts = Counter()
    for s_info in audit_data["splits"].values():
        for k, v in s_info["category_distribution"].items():
            total_categories[k] += v
        for k, v in s_info["script_distribution"].items():
            total_scripts[k] += v

    audit_data["overall_category_distribution"] = dict(total_categories)
    audit_data["overall_script_distribution"] = dict(total_scripts)
    audit_data["elapsed_audit_seconds"] = round(time.time() - start_time, 2)

    # Save JSON manifest
    manifest_dir = root_dir / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    json_path = manifest_dir / "medocr_vision_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    print(f"[+] Audit complete in {audit_data['elapsed_audit_seconds']}s! JSON saved to {json_path}")
    return audit_data, str(json_path)


if __name__ == "__main__":
    audit_medocr_dataset()
