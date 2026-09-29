# Prescription OCR Training & Evaluation Report (Phase 4)

**Model / Pipeline**: Prescription OCR Feature Extractor & Field Parser  
**Dataset**: MedOCR-Vision (Verified Prescription & Medical Report Subset)  
**Dataset SHA-256**: `664201c9bb855547e626d8edcad4101da0eef113762606400284330279ef5e7b`  
**Checkpoint Path**: [`checkpoints\phase4\ocr\best_ocr_extractor.pt`](file:///C:/Users/AIML/Documents/clg mc/checkpoints/phase4/ocr/best_ocr_extractor.pt)  
**Report JSON**: [`reports/training/phase4/ocr/ocr_report.json`](file:///C:/Users/AIML/Documents/clg mc/reports/training/phase4/ocr/ocr_report.json)

---

## 1. Domain-Isolated Evaluation Methodology

In accordance with Phase 3 audit findings, MedOCR-Vision contains ~38.9% retail receipts. The OCR evaluation pipeline strictly isolates the medical prescription and laboratory report subsets to evaluate clinical performance accurately.

---

## 2. Quantitative Performance Metrics

| Metric | Result | Target Benchmark |
| :--- | :--- | :--- |
| **Character Error Rate (CER)** | **0.00%** | < 15.0% |
| **Word Error Rate (WER)** | **0.00%** | < 20.0% |
| **Field Extraction Accuracy** | **83.33%** | > 90.0% |

---

## 3. Sample Evaluations & Key-Value Parsing

- **medocr_test_rx_001** (prescription): CER=0.0, WER=0.0, Doctor='Dr. C. Rossi', Meds Extracted=1
- **medocr_test_rx_002** (prescription): CER=0.0, WER=0.0, Doctor='Dr. A. Smith', Meds Extracted=1
- **medocr_test_rx_003** (medical_document): CER=0.0, WER=0.0, Doctor='None', Meds Extracted=0
