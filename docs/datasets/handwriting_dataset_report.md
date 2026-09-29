# Doctor's Handwritten Prescription BD Dataset Audit Report

**Dataset**: Doctor's Handwritten Prescription BD Dataset  
**Task**: Handwritten Medicine Word Recognition (Pipeline B)  
**Source**: Mendeley Data / Academic Repository (`doi: 10.17632/m97z2x6239.1`)  
**License**: `CC BY 4.0` (Permitted for Academic & Research Machine Learning)  
**Calculated Hash**: `4d0013b3ec7987f05c77ce9202f3990820bcb0aa8f27e85710205b776f0cb33b`  
**Raw Path**: `data/raw/bd_handwritten/` (Preserved Intact)  
**Manifest Path**: [`data/manifests/handwriting_dataset.json`](file:///c:/Users/AIML/Documents/clg%20mc/data/manifests/handwriting_dataset.json)

---

## 1. Dataset Overview

The Doctor's Handwritten Prescription BD dataset contains segmented word image crops extracted from authentic handwritten medical prescriptions from clinical practices. Each image corresponds to a handwritten medicine brand name, labeled with its transcribed brand name and its active generic drug category.

- **Total Images**: **4,680**
- **Unique Medicine Brand Classes**: **78**
- **Unique Generic Classes**: **15**
- **Data Integrity**: **100% Valid** (4680 images verified, 0 corrupt, 0 missing).

---

## 2. Split Analysis & Partition Statistics

| Split | Sample Count | Class Count | Samples / Class | Image Width (Mean ± Min/Max) | Image Height (Mean ± Min/Max) | Color Modes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Training** | 3,120 (66.7%) | 78 | Exactly 40 / class | 160.9 px (25 - 980) | 53.9 px (12 - 266) | RGBA (1220), L (1302), RGB (597), P (1) |
| **Validation** | 780 (16.7%) | 78 | Exactly 10 / class | 157.2 px (43 - 1251) | 52.5 px (12 - 306) | L (408), RGB (245), RGBA (127) |
| **Testing** | 780 (16.7%) | 78 | Exactly 10 / class | 163.0 px (43 - 1298) | 55.8 px (13 - 293) | RGBA (499), L (238), RGB (43) |
| **Total** | **4,680 (100%)** | **78** | **60 / class total** | **160.6 px** | **54.0 px** | **Mixed RGB / RGBA / Grayscale** |

> [!NOTE]
> The split structure is strictly balanced: every single one of the 78 medicine classes contains **40 training instances, 10 validation instances, and 10 testing instances**.

---

## 3. Image Characteristics & Preprocessing Requirements

1. **Color Modes**:
   - The images have heterogeneous PIL color modes (`L`, `RGB`, `RGBA`, `P`).
   - **Preprocessing Requirement**: All images must be converted to standard RGB or grayscale (`L`) with white-background alpha-flattening before tensor transformation.
2. **Aspect Ratios**:
   - Widths range from 25px to 1,298px, with mean ~160px.
   - Heights range from 12px to 306px, with mean ~54px.
   - Standard fixed-height resizing (e.g. 64x256 or 32x128) with padding is required for CRNN / CTC loss computation.

---

## 4. Duplicate & Cross-Split Leakage Analysis

- **Exact Image Hash Uniqueness**: 4,406 unique hashes across 4,680 files.
- **Within-Split Duplication**: 187 duplicate hash pairs found within individual splits (due to multiple identical crops or digitizations of recurring template tokens).
- **Cross-Split Duplicates**: **55 exact image hash collisions** detected across splits.
  - *Recommendation*: Preserve the official test partition boundaries during benchmarking, but flag cross-split image collisions for deduplicated evaluation.

---

## 5. Medicine Classes & Generic Formulations (Sample)

| Medicine Brand Name | Generic Name | Sample Count (Train/Val/Test) |
| :--- | :--- | :--- |
| **Aceta** | Paracetamol | 40 / 10 / 10 |
| **Ace** | Paracetamol | 40 / 10 / 10 |
| **Alatrol** | Cetirizine Dihydrochloride | 40 / 10 / 10 |
| **Amodis** | Metronidazole | 40 / 10 / 10 |
| **Azithral** / **Azithrocin** | Azithromycin | 40 / 10 / 10 |
| **Ciprocin** | Ciprofloxacin | 40 / 10 / 10 |
| **Dolo** / **Fast** | Paracetamol | 40 / 10 / 10 |
| **Napa** | Paracetamol | 40 / 10 / 10 |
| **Omep** / **Seclo** | Omeprazole | 40 / 10 / 10 |
| **Pantonix** | Pantoprazole | 40 / 10 / 10 |

---

## 6. Suitability & Role Assessment

- **Role**: Primary ground truth and benchmark for **Handwritten Medicine Word Recognition (Pipeline B)**.
- **Strengths**: Authentic cursive physician handwriting from clinical prescriptions, multi-class balanced layout, labeled with both brand and generic names.
- **Limitations**: Word-level crops only (does not provide full-page paragraph layout or bounding box hierarchy). Requires pairing with layout parser (FUNSD / synthetic) for end-to-end full prescription reading.
