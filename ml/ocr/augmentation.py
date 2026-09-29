"""Synthetic prescription image generation and data augmentation engine.

Strictly adheres to:
- Approved structured clinical vocabulary (RxNorm/RxTerms)
- Explicit synthetic specimen watermarking and disclaimers
- Zero real patient identifying information
- Deterministic seed-based generation
- Machine-readable structured annotations with bounding boxes and field hierarchy
"""
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image, ImageDraw, ImageFont, ImageFilter


class SyntheticPrescriptionGenerator:
    """Procedural generator for realistic, compliant synthetic medical prescriptions."""

    SYNTHETIC_CLINICS = [
        {"name": "Simulated Medical Research Clinic #1", "reg": "SYNTH-MED-001", "doctor": "Dr. Synthetic Specimen A, MD (Internal Med)"},
        {"name": "Simulated Clinical Testing Lab #2", "reg": "SYNTH-MED-002", "doctor": "Dr. Synthetic Specimen B, MD (Pulmonology)"},
        {"name": "Simulated Healthcare Facility #3", "reg": "SYNTH-MED-003", "doctor": "Dr. Synthetic Specimen C, MD (Cardiology)"},
        {"name": "Simulated Diagnostic Dispensary #4", "reg": "SYNTH-MED-004", "doctor": "Dr. Synthetic Specimen D, MD (Pediatrics)"},
    ]

    SYNTHETIC_PATIENTS = [
        {"id": "SYNTH_PAT_001", "name": "Synthetic Patient Alpha", "age": 32, "gender": "Unspecified"},
        {"id": "SYNTH_PAT_002", "name": "Synthetic Patient Beta", "age": 48, "gender": "Unspecified"},
        {"id": "SYNTH_PAT_003", "name": "Synthetic Patient Gamma", "age": 61, "gender": "Unspecified"},
        {"id": "SYNTH_PAT_004", "name": "Synthetic Patient Delta", "age": 25, "gender": "Unspecified"},
        {"id": "SYNTH_PAT_005", "name": "Synthetic Patient Epsilon", "age": 54, "gender": "Unspecified"},
    ]

    FREQUENCIES = [
        {"code": "1-0-1 (BD)", "desc": "Twice daily after meals"},
        {"code": "1-0-0 (OD)", "desc": "Once daily morning"},
        {"code": "0-0-1 (HS)", "desc": "Once daily at bedtime"},
        {"code": "1-1-1 (TDS)", "desc": "Three times daily with food"},
        {"code": "SOS", "desc": "As needed for symptoms"},
    ]

    DURATIONS = ["3 days", "5 days", "7 days", "10 days", "14 days", "30 days"]

    INSTRUCTIONS = [
        "Take plenty of fluids and maintain adequate hydration.",
        "Take medications after meals unless otherwise specified.",
        "Review at clinic if symptoms do not improve within 5 days.",
        "Maintain rest and monitor vital signs as advised.",
    ]

    WATERMARK_TEXT = "*** SYNTHETIC SPECIMEN - NOT A VALID PRESCRIPTION - FOR TESTING ONLY ***"

    def __init__(
        self,
        output_dir: Path,
        vocab_path: Optional[Path] = None,
        width: int = 800,
        height: int = 1000
    ):
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height

        # Load approved structured vocabulary
        if vocab_path is None:
            root_dir = Path(__file__).resolve().parent.parent.parent
            vocab_path = root_dir / "data" / "processed" / "normalization" / "normalized_medicines.json"

        self.vocab_path = Path(vocab_path)
        self.approved_medicines = self._load_approved_vocabulary()

    def _load_approved_vocabulary(self) -> List[dict]:
        """Load approved medicines from normalized RxNorm knowledge base."""
        if self.vocab_path.exists():
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        # Fallback structured medicines with RxCUI
        return [
            {"rxcui": "308189", "name": "Amoxicillin 500 MG Oral Tablet", "ingredient": "Amoxicillin", "strength": "500 mg", "aliases": ["Amoxil", "Novamox"]},
            {"rxcui": "161", "name": "Acetaminophen 650 MG Oral Tablet", "ingredient": "Paracetamol", "strength": "650 mg", "aliases": ["Dolo 650", "Calpol", "Crocin"]},
            {"rxcui": "198440", "name": "Azithromycin 500 MG Oral Tablet", "ingredient": "Azithromycin", "strength": "500 mg", "aliases": ["Azithral", "Azee"]},
            {"rxcui": "860975", "name": "Metformin hydrochloride 500 MG Oral Tablet", "ingredient": "Metformin", "strength": "500 mg", "aliases": ["Glycomet"]},
            {"rxcui": "310489", "name": "Cetirizine hydrochloride 10 MG Oral Tablet", "ingredient": "Cetirizine", "strength": "10 mg", "aliases": ["Cetzine", "Okacet"]},
            {"rxcui": "312615", "name": "Pantoprazole 40 MG Delayed Release Oral Tablet", "ingredient": "Pantoprazole", "strength": "40 mg", "aliases": ["Pan 40", "Pantocid"]},
            {"rxcui": "310965", "name": "Ibuprofen 400 MG Oral Tablet", "ingredient": "Ibuprofen", "strength": "400 mg", "aliases": ["Brufen", "Ibugesic"]},
        ]

    def generate_single_prescription(self, index: int, split: Optional[str] = None, seed: int = 42) -> Tuple[Image.Image, dict]:
        """Generate a single deterministic synthetic prescription image and machine-readable metadata."""
        split_name = split or "train"
        split_offset = {"train": 10000, "val": 20000, "validation": 20000, "test": 30000}.get(split_name, 0)
        rng = random.Random(seed + split_offset + index)

        clinic = rng.choice(self.SYNTHETIC_CLINICS)
        patient = rng.choice(self.SYNTHETIC_PATIENTS)
        day = rng.randint(1, 28)
        month = rng.randint(1, 12)
        date_str = f"2026-{month:02d}-{day:02d}"

        num_meds = rng.randint(2, 4)
        selected_concepts = rng.sample(self.approved_medicines, min(num_meds, len(self.approved_medicines)))

        inst = rng.choice(self.INSTRUCTIONS)

        # Create paper canvas with subtle texture
        bg_color = (rng.randint(250, 255), rng.randint(250, 255), rng.randint(245, 252))
        img = Image.new("RGB", (self.width, self.height), bg_color)
        draw = ImageDraw.Draw(img)

        # Top Synthetic Disclaimer Banner
        draw.rectangle([10, 10, self.width - 10, 32], fill=(240, 230, 230), outline=(200, 100, 100), width=1)
        draw.text((25, 14), self.WATERMARK_TEXT, fill=(180, 40, 40))

        # Clinic Header Box
        draw.rectangle([20, 42, self.width - 20, 125], outline=(160, 160, 180), width=1)
        draw.text((40, 50), clinic["name"], fill=(20, 30, 80))
        draw.text((40, 72), clinic["doctor"], fill=(50, 50, 50))
        draw.text((40, 95), f"Registration: {clinic['reg']}", fill=(90, 90, 90))
        draw.text((self.width - 220, 95), f"Date: {date_str}", fill=(50, 50, 50))

        # Synthetic Patient Information Bar
        draw.line([20, 138, self.width - 20, 138], fill=(180, 180, 180), width=2)
        draw.text((40, 148), f"Patient: {patient['name']} (ID: {patient['id']})", fill=(30, 30, 30))
        draw.text((480, 148), f"Age: {patient['age']} Y / Gender: {patient['gender']}", fill=(30, 30, 30))
        draw.line([20, 172, self.width - 20, 172], fill=(180, 180, 180), width=1)

        # Rx Symbol
        draw.text((40, 185), "Rx", fill=(10, 30, 140))

        # Prescription Medicine Items
        y_cursor = 225
        med_bboxes = []
        for i, concept in enumerate(selected_concepts, 1):
            freq = rng.choice(self.FREQUENCIES)
            dur = rng.choice(self.DURATIONS)

            # Choose either standard generic name or recognized trade alias
            if concept.get("aliases") and rng.random() > 0.4:
                chosen_brand = rng.choice(concept["aliases"])
                display_med_name = f"{chosen_brand} ({concept['strength']})"
            else:
                display_med_name = f"{concept['ingredient']} {concept['strength']}"

            med_line = f"{i}. {display_med_name}  ---  {freq['code']}  x  {dur}"
            draw.text((55, y_cursor), med_line, fill=(20, 20, 20))
            draw.text((80, y_cursor + 24), f"Advice: {freq['desc']}", fill=(90, 90, 90))

            med_bboxes.append({
                "item_index": i,
                "raw_text": display_med_name,
                "canonical_name": concept["name"],
                "rxcui": concept["rxcui"],
                "ingredient": concept["ingredient"],
                "strength": concept["strength"],
                "frequency": freq["code"],
                "duration": dur,
                "bbox": [55, y_cursor, self.width - 60, y_cursor + 45]
            })
            y_cursor += 65

        # Instructions / Advice Section
        draw.line([20, y_cursor + 20, self.width - 20, y_cursor + 20], fill=(200, 200, 200), width=1)
        draw.text((40, y_cursor + 32), "General Instructions:", fill=(50, 50, 50))
        draw.text((60, y_cursor + 55), f"- {inst}", fill=(70, 70, 70))

        # Bottom Disclaimer Watermark
        draw.text((60, self.height - 120), "SPECIMEN GENERATED FOR MACHINE LEARNING TESTING - CONTAINS NO REAL PHI", fill=(160, 160, 160))

        # Doctor Signature Placeholder
        sig_y = self.height - 90
        draw.text((self.width - 280, sig_y), "Authorized Digital Signature:", fill=(80, 80, 80))
        draw.line([self.width - 280, sig_y + 35, self.width - 40, sig_y + 35], fill=(80, 80, 80), width=1)
        draw.text((self.width - 270, sig_y + 40), clinic["doctor"].split(",")[0], fill=(120, 120, 120))

        # Subtle noise / blur
        if rng.random() > 0.6:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.25))

        sample_id = f"synth_rx_{split}_{index:04d}" if split else f"synth_rx_{index:04d}"
        annotation = {
            "prescription_id": sample_id,
            "split": split_name,
            "seed_used": seed + split_offset + index,
            "is_synthetic": True,
            "disclaimer": self.WATERMARK_TEXT,
            "doctor": clinic,
            "patient": patient,
            "date": date_str,
            "medicines": med_bboxes,
            "instructions": [inst],
            "image_filename": f"{sample_id}.png"
        }

        return img, annotation

    def generate_batch(self, count: int = 1000, seed: int = 42) -> List[dict]:
        """Generate full synthetic dataset batch."""
        images_dir = self.output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        all_annotations = []
        print(f"[*] Generating {count} synthetic prescriptions with seed={seed}...")

        for i in range(count):
            img, ann = self.generate_single_prescription(i, seed=seed)
            img_path = images_dir / ann["image_filename"]
            img.save(img_path, "PNG", optimize=True)
            all_annotations.append(ann)

        ann_file = self.output_dir / "annotations.json"
        with open(ann_file, "w", encoding="utf-8") as f:
            json.dump(all_annotations, f, indent=2)

        print(f"[+] Successfully generated {count} synthetic prescription images and annotations.")
        return all_annotations

    def generate_split_batch(
        self,
        split: str,
        count: int,
        seed: int = 42
    ) -> List[dict]:
        """Generate an isolated split batch of synthetic prescriptions."""
        split_dir = self.output_dir / split
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        annotations = []
        print(f"[*] Generating {count} synthetic prescriptions for split='{split}' (seed={seed})...")

        for i in range(count):
            img, ann = self.generate_single_prescription(i, split=split, seed=seed)
            img_path = images_dir / ann["image_filename"]
            img.save(img_path, "PNG", optimize=True)
            annotations.append(ann)

        ann_file = split_dir / "annotations.json"
        with open(ann_file, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2)

        print(f"[+] Saved {count} samples to {split_dir}")
        return annotations

