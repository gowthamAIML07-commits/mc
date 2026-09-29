# Handwritten Medicine Recognition Training Report (Phase 4)

**Model**: CRNN (CNN + Bidirectional LSTM + Linear Head)  
**Dataset**: Doctor's Handwritten Prescription BD (`doi: 10.17632/m97z2x6239.1`)  
**Dataset SHA-256**: `4d0013b3ec7987f05c77ce9202f3990820bcb0aa8f27e85710205b776f0cb33b`  
**Training Seed**: `42`  
**Checkpoint Path**: [`checkpoints\phase4\handwriting\best_handwriting_crnn.pt`](file:///C:/Users/AIML/Documents/clg mc/checkpoints/phase4/handwriting/best_handwriting_crnn.pt)  
**Report JSON**: [`reports/training/phase4/handwriting/handwriting_report.json`](file:///C:/Users/AIML/Documents/clg mc/reports/training/phase4/handwriting/handwriting_report.json)

---

## 1. Architecture & Hyperparameters

- **Feature Extractor**: 4-Block CNN with BatchNorm and ReLU (channels: 3 -> 32 -> 64 -> 128 -> 256)
- **Sequence Encoder**: 2-Layer Bidirectional LSTM (`hidden_dim=128`, `dropout=0.25`)
- **Total Parameters**: **1,222,862**
- **Optimizer**: AdamW (`lr=0.001`, `weight_decay=1e-4`) with Cosine Annealing
- **Batch Size**: 64
- **Epochs**: 10 (Best Epoch: **9**)
- **Training Duration**: 251.8 seconds

---

## 2. Training, Validation & Test Performance

| Metric | Training Split (3,120) | Validation Split (780) | Official Test Split (780) |
| :--- | :--- | :--- | :--- |
| **Loss** | 2.9165 | 2.9330 | **3.8054** |
| **Top-1 Accuracy** | 23.17% | 28.85% | **14.74%** |
| **Top-5 Accuracy** | — | 59.36% | **35.26%** |

---

## 3. Class-Level Performance & Error Breakdown

- **Highest Accuracy Classes**:
  - `Az`: 100.0% accuracy
  - `Baclofen`: 70.0% accuracy
  - `Metro`: 70.0% accuracy
  - `Napa Extend`: 70.0% accuracy
  - `Omastin`: 70.0% accuracy
- **Lowest Accuracy Classes (Challenging Cursive Strokes)**:
  - `Sergel`: 0.0% accuracy
  - `Telfast`: 0.0% accuracy
  - `Tridosil`: 0.0% accuracy
  - `Trilock`: 0.0% accuracy
  - `Vifas`: 0.0% accuracy

---

## 4. Known 55 Duplicate Hashes Impact

The dataset contains 55 exact duplicate image hashes across splits. In accordance with official benchmark protocol, all samples remained strictly in their designated splits. Due to the 55 collisions, up to 55/780 test samples (7.05%) represent identical image crops seen during training or validation, which slightly elevates baseline test recall.
