# Dataset Registry, Licensing & Multi-Task ML Mapping Specification

## 1. Overview & Licensing Policy

The **AI Medicine Assistant** employs a multi-task machine learning approach where each dataset is strictly isolated to its designated clinical or vision task. Datasets are **never indiscriminately concatenated** into a single monolithic training corpus.

### Strict Data Governance & Compliance Rules
1. **License Verification**: Every dataset must have its license, redistribution terms, and commercial permissions verified prior to ingestion.
2. **Download Safety Guard**: No dataset with unresolved licensing will be downloaded or ingested into the pipeline. Datasets requiring special licensing agreements (such as NLM UTS licensing for RxNorm) require explicit manual verification.
3. **Git Commit Prohibition**: Raw and processed medical datasets are strictly excluded via `.gitignore` and must never be committed to source control.
4. **Synthetic Segregation**: Synthetic and augmented data are explicitly tagged and kept separate from real clinical evaluation benchmarks.
5. **Knowledge vs. Training Distinctions**: RxNorm and RxTerms are designated as **Structured Knowledge Bases and Autocomplete Ontologies**, not traditional vision training corpora.

---

## 2. Dataset Registry & Ingestion Matrix

| ID | Dataset Name | Source / URL | Version / Date | License | Role / Task Category | Primary Task & Pipeline Role | Limitations & Governance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DS-1** | **Indian Medical Prescription OCR** | [Mendeley Data](https://mendeley.com/datasets/indian-prescription-ocr) | v1.0 (2022) | CC BY 4.0 | **`training`** (Vision OCR) | **Prescription OCR & Field Extraction**: Full-page prescription transcription, patient/doctor metadata parsing, Indian prescription format handling. | Requires author attribution. Must not commit raw clinical PII. Regional Indian formats only. |
| **DS-2** | **Doctor's Handwritten Prescription BD** | [Kaggle Research](https://www.kaggle.com/datasets/doctor-handwritten-prescription-bd) | v1.0 (~4,680 images) | CC BY 4.0 | **`training`** (Handwriting) | **Handwritten Medicine Recognition**: Cropped medicine word classification, character sequence recognition, handwriting robustness benchmarking. | Academic research and development attribution required. Contains isolated word tokens. |
| **DS-3** | **RxNorm** | [NLM / NIH](https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html) | Monthly Release (2024) | UMLS Metathesaurus License | **`knowledge_base`** (Ontology) | **Knowledge Base & Normalization**: Standardized drug nomenclature, RxCUI concept mapping, active ingredient resolution, strength normalization. | **Manual Review Required**: Requires free UTS license agreement. Raw database files must not be publicly redistributed. International restrictions apply per UMLS terms. |
| **DS-4** | **RxTerms** | [NLM RxTerms](https://mor.nlm.nih.gov/RxTerms/) | 2024 Current | Open Access Public Health Data (NLM) | **`autocomplete`** (Search Index) | **Prescription Terminology & Autocomplete**: Search indexing, fast prescription-oriented autocomplete, dosage route standardization. | Provided for clinical interface terminology by NLM. Not a substitute for full clinical decision support systems. |
| **DS-5** | **FUNSD (Form Understanding in Noisy Scanned Documents)** | [ICDAR / EPFL](https://guillaumejaume.github.io/FUNSD/) | v1.0 (199 annotated forms) | Academic / Research Open Access | **`layout_pretraining`** (Layout) | **Layout Analysis & Key-Value Parsing**: Pretraining layout transformer backbones for semantic field linking (Key $\rightarrow$ Value). | **Manual Review Required**: Academic evaluation and pretraining for document layout understanding only. Commercial deployment requires clean-room re-annotation. |
| **DS-6** | **Synthetic Prescription Dataset** | Internal Generator (`scripts/generate_synthetic_rx.py`) | v1.0 (1,000 synthetic samples) | MIT License | **`augmentation`** (Prototyping) | **Data Augmentation & Prototyping**: Cold-start testing, layout variation robustness, OCR baseline validation without privacy risk. | Synthetic data for bootstrapping, layout robustness, and augmentation testing. Never used as real clinical test benchmark. |

---

## 3. Multi-Task ML Pipeline Mapping

```
                                      +---------------------------------------------+
                                      |              RAW DATASETS                   |
                                      +---------------------+-----------------------+
                                                            |
                     +---------------------+----------------+--------------------+---------------------+
                     |                     |                                     |                     |
                     v                     v                                     v                     v
              +--------------+      +--------------+                      +--------------+      +--------------+
              |     DS-1     |      |     DS-2     |                      |  DS-3 & DS-4 |      |     DS-5     |
              | Indian Presc |      | BD Handwr.   |                      | RxNorm/Terms |      |    FUNSD     |
              +------+-------+      +------+-------+                      +------+-------+      +------+-------+
                     |                     |                                     |                     |
                     | Full Scans          | Cropped Words                       | Drug Ontologies     | Layout Forms
                     v                     v                                     v                     v
       +-----------------------+ +-----------------------+        +-----------------------+ +-----------------------+
       | Multi-Engine OCR      | | Handwritten Word      |        | Deterministic DB &    | | Document Layout       |
       | & Document Model      | | Recognition Model     |        | Normalizer Service    | | Understanding Engine  |
       | (TrOCR / PaddleOCR)   | | (CRNN + CTC / TrOCR)  |        | (PostgreSQL + Search) | | (LayoutLMv3 / Donut)  |
       +-----------+-----------+ +-----------+-----------+        +-----------+-----------+ +-----------+-----------+
                   |                         |                                |                         |
                   +------------+------------+                                |                         |
                                |                                             |                         |
                                v                                             v                         v
                   +------------------------------------------------------------------------------------+
                   |                         UNIFIED EXTRACTION & VERIFICATION ENGINE                   |
                   |      Prescription JSON -> Confidence Scoring -> Human-in-the-Loop Review           |
                   +------------------------------------------------------------------------------------+
```

### 3.1 Task 1: Full Document OCR & Field Extraction (DS-1 & DS-6)
* **Input**: Unconstrained prescription full image.
* **Role**: `training` & `augmentation`.
* **Target Output**: Structured JSON containing header metadata (Doctor, Hospital, Patient Age, Date) and tabular medicine records.
* **Pipeline Component**: `ml/ocr/` and `ml/ner/`.

### 3.2 Task 2: Handwritten Medicine Recognition (DS-2)
* **Input**: Cropped word-level bounding boxes detected from the prescription Rx section.
* **Role**: `training` (Vision classification / CTC).
* **Target Output**: Raw transcription text and top-5 alternative character sequence candidates.
* **Pipeline Component**: `ml/handwriting/`.

### 3.3 Task 3: Medicine Normalization & Identity Verification (DS-3 & DS-4)
* **Input**: Noisy OCR string (e.g., `Amoxcillin 500mg tab`).
* **Role**: `knowledge_base` & `autocomplete`.
* **Target Output**: Normalized Generic Name (`Amoxicillin`), Brand Name (`Amoxil`), RxCUI (`308189`), Ingredient (`Amoxicillin`), Strength (`500 mg`), Route (`Oral`).
* **Pipeline Component**: `ml/embeddings/` + Deterministic PostgreSQL / Redis Matcher.

### 3.4 Task 4: Layout & Key-Value Understanding (DS-5)
* **Input**: Document tokens and 2D bounding boxes.
* **Role**: `layout_pretraining`.
* **Target Output**: Semantic relation graphs linking `Header -> Value` (e.g., "Dr. Name:" -> "Dr. R. K. Sharma").
* **Pipeline Component**: Document Layout Parser.

---

## 4. Verification & Governance Summary

* **Permissive & Direct Access**:
  * DS-1 (Indian Rx OCR) - CC BY 4.0
  * DS-2 (BD Handwritten) - CC BY 4.0
  * DS-4 (RxTerms) - NLM Open Access
  * DS-6 (Synthetic Prescriptions) - MIT License
* **Requires Manual License Acceptance / UTS Verification**:
  * DS-3 (RxNorm) - UMLS Metathesaurus License (Free UTS Account required; direct automated scraping prohibited).
  * DS-5 (FUNSD) - Academic / Research Only License (Commercial deployment restricted).
