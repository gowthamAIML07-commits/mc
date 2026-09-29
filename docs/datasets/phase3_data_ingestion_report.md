# Phase 3 State Verification & Data Ingestion / Preprocessing Report

**Project**: AI Medicine Assistant  
**Task**: Phase 3 State Verification + Multi-Task Data Ingestion and Preprocessing Setup  
**Date**: 2026-09-29  
**Auditor / Engineer**: Antigravity Principal ML & Data Architecture  
**Test Suite Status**: 42 / 42 Tests Passing (100% Green)

---

## 1. Actual Current Repository State & Reconciliation

An exhaustive state audit of the repository was conducted to reconcile contradictory statements from previous execution reports:

### State Reconciliation Summary
- **Model Training Status**: **FORMAL PRODUCTION TRAINING HAS NOT STARTED**.
  - Preliminary prototype / smoke-test scripts (`training/pipelines/train_*.py`) were previously run on mock fixtures and micro-batches to validate interface signatures, generating small demo artifacts (`checkpoints/*.pt` and `reports/training/*.json`).
  - **Reconciliation**: Real full-scale model training (CRNN handwriting recognizer, OCR ViT, layout parser) has **NOT** been performed. The repository is at **Phase 3 (Data Ingestion, Verification, and Preprocessing Setup)**.
- **Raw Data Immutability**: All downloaded raw datasets under `data/raw/` remain 100% untouched and read-only.
- **Dataset Manifests & Provenance**: All local datasets now have complete, machine-readable manifests under `data/manifests/`.

---

## 2. Dataset Sources, Hashes, and Sizes

| Dataset Name | Source Authority / URL | Raw Local Path | SHA-256 Hash | Size / Total Samples |
| :--- | :--- | :--- | :--- | :--- |
| **Indian Medical Prescription OCR (MedOCR-Vision)** | [Hugging Face (`naazimsnh02/medocr-vision-dataset`)](https://huggingface.co/datasets/naazimsnh02/medocr-vision-dataset) | [`data/raw/indian_medical_prescription_ocr/`](file:///c:/Users/AIML/Documents/clg%20mc/data/raw/indian_medical_prescription_ocr) | `664201c9bb855547e626d8edcad4101da0eef113762606400284330279ef5e7b` | **2,462 samples** (Arrow format) |
| **Doctor's Handwritten Prescription BD** | [Mendeley Data (`doi:10.17632/m97z2x6239.1`)](https://data.mendeley.com/datasets/m97z2x6239/1) | [`data/raw/bd_handwritten/`](file:///c:/Users/AIML/Documents/clg%20mc/data/raw/bd_handwritten) | `9622d103fa7d10e5270c538cbcf7564d6db3a3fa1e4dfad0b0d39e8c4e477610` | **4,680 images** (78 classes) |
| **RxNorm + RxTerms Knowledge Base** | [U.S. National Library of Medicine (NIH)](https://www.nlm.nih.gov/research/umls/rxnorm/) | `data/processed/normalization/` | `0f507ee64dc17cb834f8287eaae38497676759fe419d854ce4a63116801991ad` | **15 Core Clinical Concepts**, 85 Aliases, Multi-Tier Index |
| **Synthetic Prescription Generator** | Procedural Engine (`ml/ocr/augmentation.py`) | `data/processed/synthetic_rx_specimen/` | Deterministic Seed (`42`) | **25 validation specimens** (15 train, 5 val, 5 test) |

---

## 3. Split Information & Partition Boundaries

### A. MedOCR-Vision (Indian Medical Prescription OCR)
- **Train**: 1,969 samples (80.0%)
- **Validation**: 246 samples (10.0%)
- **Test**: 247 samples (10.0%)
- **Total**: 2,462 samples

### B. Doctor's Handwritten Prescription BD
- **Training**: 3,120 image crops (78 unique medicine classes, exactly 40 samples per class)
- **Validation**: 780 image crops (78 unique medicine classes, exactly 10 samples per class)
- **Testing**: 780 image crops (78 unique medicine classes, exactly 10 samples per class)
- **Total**: 4,680 image crops (60 samples per class across 78 classes)

### C. Synthetic Prescription Specimen Dataset
- **Train**: 15 synthetic specimens (Seed offset `10000`)
- **Val**: 5 synthetic specimens (Seed offset `20000`)
- **Test**: 5 synthetic specimens (Seed offset `30000`)
- **Total**: 25 generated validation samples

---

## 4. Leakage Findings & Data Quality Analysis

1. **MedOCR-Vision Leakage**:
   - **Cross-Split Image Duplication**: 1 exact duplicate between `train` and `val`, 1 exact duplicate between `train` and `test`.
   - **Semantic / Template Leakage**: Identical doctor names and synthetic clinic templates repeated across train and test partitions.
   - **Domain Mix**: 38.9% retail receipts, 44.6% synthetic templated prescriptions, 14.9% hospital lab reports.
2. **Doctor's Handwritten Prescription BD Leakage**:
   - **Cross-Split Collisions**: 55 image hash collisions across splits (arising from identical word crop extractions of recurring clinical terms).
   - **Data Health**: 0 missing images, 0 corrupt images. Balanced across 78 classes.

---

## 5. License & Upstream Provenance Status

- **MedOCR-Vision**: Publisher declared `MIT`, but dataset incorporates commercial retail receipts and Indian hospital lab reports. **Status**: `REQUIRES MANUAL REVIEW / PROVENANCE RESTRICTION`.
- **Doctor's Handwritten Prescription BD**: `CC BY 4.0` (Permitted for academic and research machine learning).
- **RxNorm**: Governed by the UMLS Metathesaurus License Agreement. Raw redistribution restricted; derived index tables permitted.
- **RxTerms**: Open Access / Public Domain by US NLM.
- **Synthetic Prescription Generator**: Internal procedural codebase (Permissive / MIT).

---

## 6. RxNorm / RxTerms Ingestion Status

- Structured multi-tier knowledge base built at [`data/processed/normalization/`](file:///c:/Users/AIML/Documents/clg%20mc/data/processed/normalization).
- **Matching Tiers**:
  1. **Tier 0**: Clinical abbreviation and shorthand resolution (e.g. `PCM` -> Paracetamol, `AMX` -> Amoxicillin).
  2. **Tier 1**: Exact string and lowercase trade alias matching (`Dolo 650`, `Pan 40`, `Augmentin`).
  3. **Tier 2**: Phonetic clustering using Double Metaphone and Soundex keys (`Amoxcillin`, `Paractaml`).
  4. **Tier 3**: Global fuzzy Levenshtein candidate search with dosage bonus.
- **Approved Vocabulary**: Exported to [`data/processed/normalization/approved_vocabulary.json`](file:///c:/Users/AIML/Documents/clg%20mc/data/processed/normalization/approved_vocabulary.json) for cross-pipeline validation.

---

## 7. Synthetic Prescription Generator Status

- Procedural generator upgraded in [`ml/ocr/augmentation.py`](file:///c:/Users/AIML/Documents/clg%20mc/ml/ocr/augmentation.py).
- **Key Features Implemented**:
  - All generated medicine items strictly linked to approved RxNorm concepts with RxCUIs, active ingredients, and strengths.
  - Visible synthetic watermarking banner: `"*** SYNTHETIC SPECIMEN - NOT A VALID PRESCRIPTION - FOR TESTING ONLY ***"`.
  - Zero Protected Health Information (PHI): Synthetic identifiers and doctor personas only.
  - Deterministic seed isolation across `train`, `val`, and `test` partitions.
  - Machine-readable bounding box hierarchy.

---

## 8. Unresolved Discrepancies

1. **Naming Ambiguity ("Indian Medical Prescription OCR")**:
   - Manifests in the community refer to `naazimsnh02/medocr-vision-dataset` as "Indian Medical Prescription OCR", but our audit proves it is ~39% commercial retail receipts and 44% templated synthetic text. We have documented this discrepancy explicitly rather than renaming or altering raw data.
2. **Prior Prototype Metrics**:
   - Metrics in `reports/training/*.json` are prototype smoke-test outputs and should not be cited as production model benchmarks.

---

## 9. Recommended Next Step

With all Phase 3 datasets ingested, audited, verified, and indexed, the project is ready for **Phase 4 Multi-Task Model Training & Evaluation**:
1. Implement and train the CRNN / TrOCR handwritten medicine recognizer on `Doctor's Handwritten Prescription BD`.
2. Configure layout parsing baseline on FUNSD and synthetic structured templates.
3. Benchmark end-to-end inference combining OCR extraction, layout association, and RxNorm normalization.
