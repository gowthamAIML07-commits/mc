# RxNorm / RxTerms Clinical Normalization Benchmark Report (Phase 4)

**Resource**: RxNorm (NLM) + RxTerms Multi-Tier Knowledge Base  
**Knowledge Base Hash**: `074909cc643919ed32b589cd341c1d67aa8f3f3d3747a17ae2d20b204791a236`  
**Report JSON**: [`reports/training/phase4/normalization/normalization_report.json`](file:///C:/Users/AIML/Documents/clg mc/reports/training/phase4/normalization/normalization_report.json)

---

## 1. Quantitative Benchmark Results

| Metric | Score | Target |
| :--- | :--- | :--- |
| **Top-1 Normalization Accuracy** | **95.45%** | > 95.0% |
| **Top-5 Candidate Recall** | **100.0%** | > 98.0% |
| **False Positive Rate** | **0.00%** | < 2.0% |
| **Unresolved Rate (Out-of-Vocab)** | **4.55%** | Safe Fallback to `REQUIRES_REVIEW` |

---

## 2. Category Performance Summary

- **Exact Canonical Terms**: 100.0% accuracy (`exact_hash` tier)
- **Trade Aliases (Dolo, Pan, Azithral, etc.)**: 100.0% accuracy (`exact_sanitized` tier)
- **Clinical Abbreviations (PCM, AMX, AZM, PANTO)**: 100.0% accuracy (`abbreviation_resolution` tier)
- **Phonetic & OCR Noise (Paractaml, Amoxcillin, etc.)**: 100.0% accuracy (`phonetic_metaphone` tier)
- **Out-of-Vocabulary Safety**: 100.0% marked `unverified` / `REQUIRES_REVIEW` (Zero hallucinated RxCUIs)
