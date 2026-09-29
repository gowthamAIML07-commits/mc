"""Dataset extraction, standardization, and seed generation script."""
import json
import os
import shutil
import zipfile
from pathlib import Path


def extract_bd_handwritten(root_dir: Path):
    """Extract Doctor's Handwritten Prescription BD dataset from Downloads."""
    downloads_dir = Path("C:/Users/AIML/Downloads")
    zip1 = downloads_dir / "archive (1).zip"
    zip2 = downloads_dir / "archive.zip"
    
    raw_target = root_dir / "data" / "raw" / "bd_handwritten"
    processed_target = root_dir / "data" / "processed" / "bd_handwritten"
    raw_target.mkdir(parents=True, exist_ok=True)
    processed_target.mkdir(parents=True, exist_ok=True)

    if zip1.exists():
        print(f"[*] Extracting {zip1.name} to {raw_target}...")
        with zipfile.ZipFile(zip1, "r") as zf:
            for member in zf.infolist():
                # Clean path encoding
                filename = member.filename.replace("’", "'")
                target_path = raw_target / filename
                if member.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
        print("[+] Extracted raw BD Handwritten dataset.")

    if zip2.exists():
        print(f"[*] Extracting processed mappings from {zip2.name} to {processed_target}...")
        with zipfile.ZipFile(zip2, "r") as zf:
            zf.extractall(processed_target)
        print("[+] Extracted processed BD Handwritten dataset.")


def prepare_rxnorm_and_rxterms(root_dir: Path):
    """Prepare curated RxNorm and RxTerms ontology seed datasets."""
    rxnorm_dir = root_dir / "data" / "processed" / "rxnorm"
    rxterms_dir = root_dir / "data" / "processed" / "rxterms"
    rxnorm_dir.mkdir(parents=True, exist_ok=True)
    rxterms_dir.mkdir(parents=True, exist_ok=True)

    # Standardized clinical medicine knowledge base
    rxnorm_concepts = [
        {"rxcui": "308189", "name": "Amoxicillin 500 MG Oral Tablet", "ingredient": "Amoxicillin", "strength": "500 mg", "form": "Oral Tablet", "class": "Penicillin Antibiotic", "aliases": ["Amoxil", "Amoxcillin", "Amoxycillin", "Moxatag"]},
        {"rxcui": "312320", "name": "Amoxicillin 250 MG Oral Capsule", "ingredient": "Amoxicillin", "strength": "250 mg", "form": "Oral Capsule", "class": "Penicillin Antibiotic", "aliases": ["Amoxil", "Amoxicillin 250"]},
        {"rxcui": "161", "name": "Paracetamol 650 MG Oral Tablet", "ingredient": "Paracetamol", "strength": "650 mg", "form": "Oral Tablet", "class": "Analgesic / Antipyretic", "aliases": ["Dolo 650", "Calpol", "Crocin", "Acetaminophen", "Paractaml", "Paracetamol 650"]},
        {"rxcui": "198440", "name": "Azithromycin 500 MG Oral Tablet", "ingredient": "Azithromycin", "strength": "500 mg", "form": "Oral Tablet", "class": "Macrolide Antibiotic", "aliases": ["Zithromax", "Azithral", "Azee 500", "Azithromicin"]},
        {"rxcui": "860975", "name": "Metformin hydrochloride 500 MG Oral Tablet", "ingredient": "Metformin", "strength": "500 mg", "form": "Oral Tablet", "class": "Biguanide Antidiabetic", "aliases": ["Glucophage", "Glycomet", "Metfornin"]},
        {"rxcui": "310965", "name": "Ibuprofen 400 MG Oral Tablet", "ingredient": "Ibuprofen", "strength": "400 mg", "form": "Oral Tablet", "class": "NSAID", "aliases": ["Advil", "Motrin", "Brufen", "Ibugesic"]},
        {"rxcui": "312615", "name": "Pantoprazole 40 MG Delayed Release Oral Tablet", "ingredient": "Pantoprazole", "strength": "40 mg", "form": "Delayed Release Tablet", "class": "Proton Pump Inhibitor", "aliases": ["Protonix", "Pan 40", "Pantocid"]},
        {"rxcui": "310489", "name": "Cetirizine hydrochloride 10 MG Oral Tablet", "ingredient": "Cetirizine", "strength": "10 mg", "form": "Oral Tablet", "class": "Antihistamine", "aliases": ["Zyrtec", "Cetzine", "Alatrol", "Cetrizine"]},
        {"rxcui": "6851", "name": "Methotrexate 10 MG Oral Tablet", "ingredient": "Methotrexate", "strength": "10 mg", "form": "Oral Tablet", "class": "Antimetabolite / Immunosuppressant", "aliases": ["Trexall", "Folitrax"]},
        {"rxcui": "855332", "name": "Atorvastatin 20 MG Oral Tablet", "ingredient": "Atorvastatin", "strength": "20 mg", "form": "Oral Tablet", "class": "HMG-CoA Reductase Inhibitor", "aliases": ["Lipitor", "Atorva", "Storvas"]},
        {"rxcui": "311689", "name": "Montelukast 10 MG Oral Tablet", "ingredient": "Montelukast", "strength": "10 mg", "form": "Oral Tablet", "class": "Leukotriene Receptor Antagonist", "aliases": ["Singulair", "Montair", "Monticope"]},
        {"rxcui": "197361", "name": "Amlodipine 5 MG Oral Tablet", "ingredient": "Amlodipine", "strength": "5 mg", "form": "Oral Tablet", "class": "Calcium Channel Blocker", "aliases": ["Norvasc", "Amlong", "Stamlo"]},
        {"rxcui": "896188", "name": "Telmisartan 40 MG Oral Tablet", "ingredient": "Telmisartan", "strength": "40 mg", "form": "Oral Tablet", "class": "Angiotensin II Receptor Blocker", "aliases": ["Micardis", "Telma 40", "Telpres"]},
        {"rxcui": "312961", "name": "Ciprofloxacin 500 MG Oral Tablet", "ingredient": "Ciprofloxacin", "strength": "500 mg", "form": "Oral Tablet", "class": "Fluoroquinolone Antibiotic", "aliases": ["Cipro", "Ciplox 500", "Ciprobid"]}
    ]

    with open(rxnorm_dir / "rxnorm_concepts.json", "w", encoding="utf-8") as f:
        json.dump(rxnorm_concepts, f, indent=2)

    # RxTerms Search Index
    rxterms_items = [
        {"displayName": item["name"], "rxtermsDoseForm": item["form"], "route": "Oral", "strength": item["strength"], "genericName": item["ingredient"], "rxcui": item["rxcui"]}
        for item in rxnorm_concepts
    ]
    with open(rxterms_dir / "rxterms_index.json", "w", encoding="utf-8") as f:
        json.dump(rxterms_items, f, indent=2)

    print(f"[+] Prepared RxNorm ({len(rxnorm_concepts)} concepts) and RxTerms ({len(rxterms_items)} terms).")


def prepare_funsd_benchmark(root_dir: Path):
    """Prepare FUNSD layout understanding annotations and key-value relations."""
    funsd_dir = root_dir / "data" / "processed" / "funsd"
    funsd_dir.mkdir(parents=True, exist_ok=True)

    funsd_samples = [
        {
            "id": "funsd_doc_001",
            "tokens": ["Dr.", "Name:", "Dr.", "Mukherjee", "Date:", "15/03/2024", "Patient:", "Aarav", "Age:", "34", "Rx:", "Amoxicillin", "500mg", "TDS", "5d"],
            "bboxes": [
                [45, 50, 65, 62], [70, 50, 110, 62], [120, 50, 145, 62], [150, 50, 230, 62],
                [400, 50, 440, 62], [445, 50, 520, 62],
                [45, 80, 100, 92], [110, 80, 160, 92],
                [200, 80, 230, 92], [235, 80, 255, 92],
                [45, 130, 75, 145], [90, 130, 180, 145], [190, 130, 240, 145], [250, 130, 280, 145], [290, 130, 310, 145]
            ],
            "relations": [
                {"head": "header_doctor", "key": "Dr. Name:", "value": "Dr. Mukherjee", "category": "doctor"},
                {"head": "header_date", "key": "Date:", "value": "15/03/2024", "category": "date"},
                {"head": "header_patient", "key": "Patient:", "value": "Aarav", "category": "patient"},
                {"head": "header_age", "key": "Age:", "value": "34", "category": "age"},
                {"head": "body_rx", "key": "Rx:", "value": "Amoxicillin 500mg TDS 5d", "category": "medicine"}
            ]
        },
        {
            "id": "funsd_doc_002",
            "tokens": ["Consultant:", "Dr.", "Patel", "Reg:", "GMC-8821", "Date:", "12/04/2024", "Patient:", "Sneha", "Age:", "28", "Rx:", "Dolo", "650", "SOS"],
            "bboxes": [
                [40, 50, 120, 62], [125, 50, 145, 62], [150, 50, 200, 62],
                [250, 50, 280, 62], [285, 50, 360, 62],
                [420, 50, 460, 62], [465, 50, 540, 62],
                [40, 85, 95, 98], [105, 85, 150, 98],
                [200, 85, 230, 98], [235, 85, 255, 98],
                [40, 140, 70, 155], [85, 140, 125, 155], [135, 140, 165, 155], [175, 140, 205, 155]
            ],
            "relations": [
                {"head": "header_doctor", "key": "Consultant:", "value": "Dr. Patel", "category": "doctor"},
                {"head": "header_reg", "key": "Reg:", "value": "GMC-8821", "category": "registration"},
                {"head": "header_date", "key": "Date:", "value": "12/04/2024", "category": "date"},
                {"head": "header_patient", "key": "Patient:", "value": "Sneha", "category": "patient"},
                {"head": "header_age", "key": "Age:", "value": "28", "category": "age"},
                {"head": "body_rx", "key": "Rx:", "value": "Dolo 650 SOS", "category": "medicine"}
            ]
        }
    ]

    with open(funsd_dir / "funsd_layout_samples.json", "w", encoding="utf-8") as f:
        json.dump(funsd_samples, f, indent=2)

    print(f"[+] Prepared FUNSD layout benchmark dataset ({len(funsd_samples)} annotated documents).")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    extract_bd_handwritten(root)
    prepare_rxnorm_and_rxterms(root)
    prepare_funsd_benchmark(root)
