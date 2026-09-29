# Machine Learning & Prescription Processing Pipeline Specification

## 1. Architectural Philosophy

The ML architecture decomposes complex clinical vision and natural language tasks into specialized, loosely-coupled, verifiable micro-models. This ensures:
1. **Explainability**: Every step (OCR -> Segmentation -> Normalization -> Verification) outputs distinct intermediate confidence metrics and bounding boxes.
2. **Deterministic Safeguards**: Probabilistic neural predictions are strictly grounded against curated ontologies (RxNorm/RxTerms) and rule-based safety firewalls before reaching the user.
3. **Graceful Degradation**: Dual-mode execution supports high-throughput GPU inference and CPU-friendly quantized fallbacks.

---

## 2. End-to-End Prescription Processing Flow

```
[IMAGE UPLOAD]
      |
[IMAGE VALIDATION] ---------> (Format, Max Size 20MB, Valid Color Channels)
      |
[IMAGE PREPROCESSING] ------> (Denoise, Grayscale, CLAHE Contrast, Deskew, Dewarp)
      |
[QUALITY & BLUR CHECK] -----> (Laplacian Variance, Brightness/Contrast Score)
      |
[DOCUMENT DETECTION] -------> (Document Contour & Perspective Correction)
      |
[OCR & LAYOUT ANALYSIS] ----> (Block Segmentation: Header, Rx Body, Footer)
      |
[FIELD EXTRACTION] ---------> (Doctor, Registration, Patient Name, Age, Date)
      |
[MEDICINE CROP DETECTION] --> (Rx Line Bounding Box Slicing)
      |
[HANDWRITING RECOGNITION] --> (CRNN / TrOCR for Isolated Words & Dosages)
      |
[MEDICINE NORMALIZATION] ---> (Phonetic Metaphone + Levenshtein + RxTerms Index)
      |
[RxNorm VERIFICATION] ------> (RxCUI Exact & Partial Concept Matching)
      |
[CONFIDENCE CALCULATION] ---> (Composite Score: OCR, Recognition, Normalization)
      |
[HUMAN REVIEW STAGE] -------> (Side-by-side Visual Diff & Interactive Editor)
      |
[FINAL PRESCRIPTION JSON] --> (Immutable Stored Record in PostgreSQL)
```

---

## 3. Image Preprocessing & Quality Assurance Subsystem

To maximize downstream OCR and handwriting recognition accuracy on low-quality smartphone captures, the preprocessing engine executes non-destructive image pipelines:

| Operation | Implementation Method | Objective |
| :--- | :--- | :--- |
| **Color Conversion** | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` | Eliminate chromatic noise and prepare single-channel data |
| **Denoising** | `cv2.fastNlMeansDenoising` / Bilateral Filter | Suppress paper texture and camera sensor grain while preserving text edges |
| **Contrast Boost** | CLAHE (Contrast Limited Adaptive Histogram Equalization) | Balance uneven lighting and flash glare across clinical paper |
| **Deskew & Orientation** | Minimum Area Bounding Rect on Hough Lines / Tesseract OSD | Correct rotation angles from -45° to +45° to 0° horizontal alignment |
| **Adaptive Thresholding** | Sauvala / Niblack & Otsu Binarization | Separate faded pen ink and pencil marks from colored prescription pads |
| **Blur Metric** | $\text{Var}(\nabla^2 \text{Image})$ (Laplacian Variance) | Quality threshold score ($S_q \in [0, 100]$); alert user if $S_q < 35$ (blurry image) |

*Storage policy*: The original pristine image is retained in immutable storage; intermediate crops and enhanced masks are generated ephemerally or stored with explicit session TTLs.

---

## 4. Multi-Model Specifications

### 4.1 Model A: Full Prescription OCR & Document Understanding
* **Primary Backbone**: PaddleOCR v4 / TrOCR Document Model + Layout Parser.
* **CPU Fallback**: Tesseract 5.0 + EasyOCR.
* **Input**: Enhanced 2D prescription scan ($1024 \times 1024$ minimum resolution).
* **Output Schema**:
  ```json
  {
    "doctor": "Dr. S. K. Mukherjee, MBBS, MD",
    "doctor_registration": "WBMC-48291",
    "patient": "Aarav Sharma",
    "patient_age": 34,
    "patient_gender": "Male",
    "date": "2024-03-15",
    "medicines_raw": ["Amoxcillin 500mg TDS 5d", "Paracetamol 650mg SOS"],
    "instructions": ["Take after food", "Plenty of fluids"],
    "layout_confidence": 0.93
  }
  ```

### 4.2 Model B: Handwritten Medicine Recognition
* **Primary Backbone**: TrOCR-handwritten / CRNN (CNN feature extractor + BiLSTM sequence encoder + CTC decoder).
* **Input**: Cropped word/line image patch from Rx bounding box ($128 \times 384 \times 3$).
* **Output Schema**:
  ```json
  {
    "candidate": "Amoxicillin",
    "confidence": 0.88,
    "alternatives": [
      {"candidate": "Ampicillin", "confidence": 0.62},
      {"candidate": "Amoxil", "confidence": 0.58}
    ]
  }
  ```

### 4.3 Model C: Medicine Normalization & Identity Verification
* **Algorithm**: 4-Tier Hybrid Matcher:
  1. *Tier 1*: Exact case-insensitive hash lookup in RxNorm active concepts.
  2. *Tier 2*: Double Metaphone phonetic key matching (handles clinical misspellings like *Amoxcilin* -> *Amoxicillin*).
  3. *Tier 3*: Normalized Levenshtein Token Sort Ratio ($\ge 85\%$).
  4. *Tier 4*: Dense semantic embedding cosine similarity using medical BGE embedding.
* **Output Schema**:
  ```json
  {
    "raw_name": "Amoxcillin 500mg",
    "normalized_name": "Amoxicillin 500 MG Oral Tablet",
    "rxnorm_id": "308189",
    "ingredient": "Amoxicillin",
    "strength": "500 mg",
    "dosage_form": "Oral Tablet",
    "confidence": 0.94,
    "verification_status": "verified"
  }
  ```

### 4.4 Model D: Clinical Query Intent Classifier
* **Backbone**: BioLinkBERT-Base / MiniLM-L6 Sequence Classifier.
* **15 Target Classes**:
  1. `MEDICINE_INFORMATION`
  2. `MEDICINE_USE`
  3. `SIDE_EFFECT`
  4. `DOSAGE_INFORMATION`
  5. `CONTRAINDICATION`
  6. `DRUG_INTERACTION`
  7. `MISSED_DOSE`
  8. `PREGNANCY`
  9. `CHILD_MEDICATION`
  10. `ELDERLY_MEDICATION`
  11. `PRESCRIPTION`
  12. `MEDICINE_IDENTIFICATION`
  13. `SYMPTOM`
  14. `EMERGENCY`
  15. `GENERAL_HEALTH`
  16. `OUT_OF_SCOPE`

### 4.5 Model E: Medical Named Entity Recognition (NER)
* **Backbone**: Bio_ClinicalBERT Token Classifier (BIO tagging scheme).
* **13 Clinical Entities**:
  `DRUG`, `BRAND`, `ACTIVE_INGREDIENT`, `DISEASE`, `SYMPTOM`, `DOSAGE`, `FREQUENCY`, `DURATION`, `AGE`, `ALLERGY`, `LAB_TEST`, `BODY_PART`, `MEDICAL_PROCEDURE`.

### 4.6 Model F: Medical Safety & Triage Classifier
* **Architecture**: Deterministic Regex Hard Rules + Multi-Head Risk Transformer.
* **Risk Levels**:
  * `EMERGENCY`: Imminent life threat (anaphylaxis, respiratory arrest, severe poisoning, chest pain radiating to arm).
  * `HIGH`: Dangerous dosage, contraindication with pregnancy, pediatric overdose risk.
  * `MODERATE`: Adverse interaction with mild symptoms, missed dose confusion.
  * `LOW`: General drug usage inquiry, storage condition query.
  * `OUT_OF_SCOPE`: Non-medical queries, financial advice, illicit synthesis.

### 4.7 Model G: Grounded RAG Generator
* **Backbone**: Configurable via `MODEL_NAME` (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, `Qwen/Qwen2.5-7B-Instruct`).
* **Format**: Generates structured, patient-friendly markdown with explicit citation anchors (`[Source: FDA Label #12]`) and standard medical disclaimers.

---

## 5. Confidence Calculation & Composite Scoring Framework

Every extracted medicine line receives a composite confidence score $C_{\text{overall}} \in [0.0, 1.0]$ computed as:

$$C_{\text{overall}} = w_{\text{ocr}} C_{\text{ocr}} + w_{\text{recog}} C_{\text{recog}} + w_{\text{norm}} C_{\text{norm}} + w_{\text{rxnorm}} \delta_{\text{verified}}$$

* Default weights: $w_{\text{ocr}} = 0.25$, $w_{\text{recog}} = 0.25$, $w_{\text{norm}} = 0.25$, $w_{\text{rxnorm}} = 0.25$.
* Verification indicator: $\delta_{\text{verified}} = 1.0$ if matching RxCUI is verified; $0.0$ otherwise.

### Verification Status Thresholds
* **`verified`**: $C_{\text{overall}} \ge 0.85$ and exact/high-confidence RxNorm match confirmed.
* **`review_required`**: $0.60 \le C_{\text{overall}} < 0.85$ (Highlighted in Amber; user must review).
* **`unverified`**: $C_{\text{overall}} < 0.60$ or no plausible RxNorm match (Highlighted in Red; requires manual correction).

---

## 6. Official Prescription JSON Schema

```json
{
  "prescription_id": "rx_987fbc82_a12e_489b",
  "patient": {
    "name": "Aarav Sharma",
    "age": 34,
    "gender": "Male"
  },
  "doctor": {
    "name": "Dr. S. K. Mukherjee",
    "registration": "WBMC-48291"
  },
  "date": "2024-03-15",
  "medicines": [
    {
      "raw_text": "Amoxcillin 500mg 1 tab TDS x 5 days",
      "normalized_name": "Amoxicillin 500 MG Oral Tablet",
      "rxnorm_id": "308189",
      "ingredient": "Amoxicillin",
      "strength": "500 mg",
      "dose": "1 tablet",
      "frequency": "3 times daily",
      "duration": "5 days",
      "route": "Oral",
      "confidence": 0.91,
      "verification_status": "verified"
    }
  ],
  "instructions": [
    "Take after meals with a full glass of water"
  ],
  "warnings": [
    "Complete full 5-day course even if symptoms resolve"
  ],
  "overall_confidence": 0.91
}
```

---

## 7. Model Evaluation Metrics

| Pipeline Stage | Evaluation Metrics | Target Production Benchmark |
| :--- | :--- | :--- |
| **Prescription OCR** | Character Error Rate (CER), Word Error Rate (WER), Field Accuracy | $\text{CER} < 8\%$, $\text{Field Acc} \ge 92\%$ |
| **Handwriting Recognition** | Top-1 Accuracy, Top-5 Accuracy, Macro F1 | $\text{Top-1} \ge 85\%$, $\text{Top-5} \ge 96\%$ |
| **Medical NER** | Token Precision, Recall, Entity F1 (Micro & Macro) | $\text{Micro F1} \ge 90\%$ |
| **Intent Classification** | Macro F1, Weighted F1, Per-class Recall | $\text{Macro F1} \ge 93\%$ |
| **Safety Classifier** | Recall on Emergency/High Risk, False Negative Rate (FNR) | $\text{Recall} \ge 99.5\%$, $\text{FNR} < 0.5\%$ |
| **RAG Retrieval** | Recall@5, Recall@10, Mean Reciprocal Rank (MRR), nDCG@10 | $\text{Recall@5} \ge 88\%$, $\text{MRR} \ge 0.85$ |
| **LLM Generation** | RAGAS Faithfulness, Answer Relevance, Hallucination Rate | $\text{Faithfulness} \ge 0.95$, $\text{Hallucination} < 2\%$ |
