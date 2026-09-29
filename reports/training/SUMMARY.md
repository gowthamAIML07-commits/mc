# Multi-Task Machine Learning Training & Evaluation Summary

**Execution Timestamp**: `2026-09-29T11:08:21Z`  
**Total Duration**: `126.61s`  
**Global Seed**: `42`  
**Hardware Platform**: `Windows 11 (Intel64 Family 6 Model 183 Stepping 1, GenuineIntel)`  

## Independent Pipeline Results Matrix

| Pipeline | Model Architecture | Dataset & Version | Key Test Metric | Checkpoint |
| :--- | :--- | :--- | :--- | :--- |
| **Pipeline A: OCR** | `PaddleOCR_TrOCR_Hybrid` | Indian Medical Prescription OCR & Synthetic Augmentation (v1.0) | **CER: 0.2375**, Field Acc: 100.0% | `checkpoints\ocr\best_ocr_extractor.pt` |
| **Pipeline B: Handwriting** | `CRNN_CNN_BiLSTM` | Doctor's Handwritten Prescription BD (v1.0) | **Top-1 Acc: 7.4399999999999995%**, Top-5: 20.77% | `checkpoints\handwriting\best_handwriting_crnn.pt` |
| **Pipeline C: Augmentation** | `Procedural_PIL_Vector_Synthesis` | Synthetic Prescription Dataset (v1.0) | **Samples Generated: 700**, Coverage: 100% | `checkpoints\augmentation\synthetic_rx_manifest.pt` |
| **Pipeline D: Layout** | `Spatial_Pair_MLP_LayerNorm` | FUNSD (Form Understanding in Noisy Scanned Documents) (v1.0) | **Key-Value Acc: 95.71%** | `checkpoints\layout\best_layout_parser.pt` |
| **Pipeline E: Normalization** | `4_Tier_Phonetic_Metaphone_Levenshtein_Embedding` | RxNorm & RxTerms Clinical Ontologies (2024 Current) | **Top-1 Match: 100.0%** | `checkpoints\normalization\rxnorm_index.pt` |
