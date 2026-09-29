# Frontend UI/UX Design System & Human-in-the-Loop Interaction Specification

## 1. Design Philosophy & Medical Aesthetics

The **AI Medicine Assistant** frontend is crafted to communicate **clinical trustworthiness, precision, and calm clarity**. Medical interfaces must never feel cluttered, chaotic, or gamified; they must prioritize readability, clear visual hierarchy, and accessible color-coding for confidence and risk indicators.

### Key Visual Principles
* **Color Palette**:
  * *Primary Slate/Navy*: Deep trustworthy blues (`#0F172A`, `#1E293B`) for navigation and structural elements.
  * *Clinical Teal/Emerald*: Soft medical accents (`#0D9488`, `#10B981`) for verified badges and positive states.
  * *Alert Amber*: Distinct warning tones (`#F59E0B`, `#D97706`) for `review_required` items and moderate interactions.
  * *Emergency Crimson*: High-visibility alert accents (`#DC2626`, `#991B1B`) strictly reserved for safety emergencies and contraindications.
  * *Neutral Backgrounds*: Clean, low-fatigue slate backgrounds (`#F8FAFC` in light mode, `#0B0F17` in dark mode).
* **Typography**:
  * Primary: `Inter` / `Geist Sans` for clean, highly legible clinical typography.
  * Monospace: `JetBrains Mono` for RxCUI codes, dosage numerals, and drug strengths.
* **Accessibility (WCAG 2.1 AAA/AA)**:
  * Minimum 4.5:1 text contrast ratio across all UI states.
  * Full keyboard navigability (`Tab`, `Enter`, `Esc`, arrow keys for image pan/zoom).
  * Explicit ARIA labels on all modal overlays, confidence badges, and interactive image bounding boxes.

---

## 2. Core Views & Page Specifications

```
                                +---------------------------+
                                |  Global Navigation / Bar  |
                                +-------------+-------------+
                                              |
      +---------------+---------------+-------+-------+---------------+---------------+
      |               |               |               |               |               |
      v               v               v               v               v               v
+-----------+   +-----------+   +-----------+   +-----------+   +-----------+   +-----------+
| Dashboard |   | Prescrip- |   | Prescrip- |   | Medicine  |   | AI Chat   |   | Drug      |
|           |   | tion      |   | tion      |   | Search &  |   | Assistant |   | Interac-  |
|           |   | Upload    |   | Review    |   | Profile   |   | (RAG)     |   | tions     |
+-----------+   +-----------+   +-----------+   +-----------+   +-----------+   +-----------+
```

### 2.1 Dashboard (`/dashboard`)
* **Welcome & Status Header**: User profile, active prescriptions count, last verified date.
* **Quick Action Cards**:
  1. *Upload Prescription* (Direct file drop area or camera capture).
  2. *Ask AI Assistant* (Instant input prompt).
  3. *Search Medicine Database* (RxNorm instant search bar).
  4. *Check Drug Interactions* (Multi-drug comparison tool).
* **Recent Activity Grid**:
  * Recent Prescriptions with thumbnail preview, extracted date, and verification status badges.
  * Recent Medicines Viewed with quick links to FDA monographs.
* **Safety Banner**: Persistent, understated reminder of clinical disclaimer and emergency contact shortcut.

### 2.2 Prescription Upload Pipeline (`/prescriptions/upload`)
* **Drag-and-Drop Dropzone**: Supports PNG, JPG, JPEG, WEBP, PDF (up to 20 MB).
* **Live Stepper & Progress Indicator**:
  1. `Upload`: Image payload validation and hashing.
  2. `Read`: Document preprocessing, deskewing, and quality scoring.
  3. `Recognize`: Layout block segmentation and handwritten crop extraction.
  4. `Verify`: RxNorm and RxTerms candidate matching and confidence calculation.
  5. `Review`: Seamless transition to the Human-in-the-Loop review screen.

---

## 3. Human-in-the-Loop Prescription Review Interface (`/prescriptions/[id]/review`)

The Review UI is the critical bridge between AI perception and patient safety. It allows users and clinicians to inspect, verify, and correct AI extractions side-by-side with visual grounding.

```
+-------------------------------------------------------------------------------------------------------+
|  Prescription #rx_987fbc82  |  Doctor: Dr. S. K. Mukherjee  |  Date: 2024-03-15  |  Status: DRAFT      |
+-------------------------------------------------------------------------------------------------------+
|                       |                                                                               |
|   LEFT PANE:          |   RIGHT PANE: Extracted Entities & Verification Controls                      |
|   Prescription Image  |                                                                               |
|   Viewer with Active  |   +-----------------------------------------------------------------------+   |
|   Bounding Boxes      |   | Medicine 1: Amoxicillin 500 mg Oral Tablet                            |   |
|                       |   | Raw OCR: "Amoxcillin 500mg"      | Confidence: 94% [Verified Badge]   |   |
|   +---------------+   |   +-----------------------------------------------------------------------+   |
|   | [Box 1: Rx]   |   |   | Strength:  [ 500 mg     ]   | Dose:      [ 1 tablet   ]              |   |
|   | "Amoxcillin"  |   |   | Frequency: [ 3x / day   ]   | Duration:  [ 5 days     ]              |   |
|   |               |   |   | Route:     [ Oral       ]   | RxCUI:     [ 308189     ]              |   |
|   | [Box 2: Rx]   |   |   +-----------------------------------------------------------------------+   |
|   | "Paracetamol" |   |   | Actions: [ Accept ]  [ Edit Fields ]  [ Search Alternative ] [ Reject]|   |
|   +---------------+   |   +-----------------------------------------------------------------------+   |
|                       |                                                                               |
|   Controls:           |   +-----------------------------------------------------------------------+   |
|   [Zoom] [Pan]        |   | Medicine 2: Paracetamol 650 mg Tablet                                 |   |
|   [Toggle Masks]      |   | Raw OCR: "Paractaml 650"         | Confidence: 71% [Review Required]  |   |
|                       |   +-----------------------------------------------------------------------+   |
|                       |   | Suggested Match: Paracetamol 650 MG Oral Tablet [Confirm Match]       |   |
|                       |   +-----------------------------------------------------------------------+   |
|                       |                                                                               |
|                       |   [ + Add Missing Medicine ]            [ Save & Finalize Prescription ]      |
+-------------------------------------------------------------------------------------------------------+
```

### Key Review Interactions
1. **Interactive Cross-Highlighting**: Hovering over or clicking a medicine card on the right automatically scrolls, pans, and highlights the corresponding bounding box crop on the prescription image.
2. **Confidence Visual Cues**:
   * Green border & check icon: `verified` ($C \ge 85\%$).
   * Amber border & warning icon: `review_required` ($60\% \le C < 85\%$).
   * Red border & alert icon: `unverified` ($C < 60\%$).
3. **Smart Autocomplete Replacement**: If the OCR misread a trade name, the user clicks "Search Alternative" to bring up instant RxTerms fuzzy search with dosage strength options.

---

## 4. AI Assistant Chat Experience (`/chat`)

* **Sidebar**: History of medical conversations, grouped by date, with search and "New Consultation" button.
* **Message Thread**:
  * **User Bubble**: Minimalist right-aligned bubble with timestamp.
  * **Assistant Message Card**:
    * Clean typography with bullet points and bold emphasis.
    * Expandable **"Sources & Clinical Citations"** accordions showing document title, publication/FDA date, and excerpt.
    * Interactive **"Medicine Chips"** embedded in text (clicking opens side drawer with dosage and warnings).
* **Suggested Queries Carousel**:
  * *"What are the common side effects of Amoxicillin?"*
  * *"Can I take Paracetamol if I have asthma?"*
  * *"Explain my prescription in simple terms"*
  * *"Check my active medications for dangerous combinations"*

---

## 5. Drug Interaction Checker UI (`/interactions`)

```
+-----------------------------------------------------------------------------------------+
|  Drug Interaction Checker (Deterministic RxNorm Knowledge Engine)                       |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  Selected Medications:                                                                  |
|  [ Amoxicillin 500mg  (x) ]  [ Methotrexate 10mg  (x) ]  [ + Add Another Drug... ]      |
|                                                                                         |
|  [ Analyze Pairwise Combinations ]                                                      |
|                                                                                         |
|  -------------------------------------------------------------------------------------  |
|  RESULTS (1 Interaction Found):                                                         |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  | [!] SEVERITY: HIGH - POTENTIAL TOXICITY WARNING                                   |  |
|  | Interaction: Amoxicillin + Methotrexate                                           |  |
|  | Clinical Description: Penicillins may decrease the renal clearance of            |  |
|  | methotrexate, resulting in elevated serum methotrexate levels and increased       |  |
|  | risk of severe hematologic and gastrointestinal toxicities.                       |  |
|  | Source: FDA DailyMed Prescribing Monograph #SPL-88210                             |  |
|  +-----------------------------------------------------------------------------------+  |
|                                                                                         |
|  DISCLAIMER: Absence of a listed interaction does not guarantee clinical compatibility. |
|  Always consult your physician or clinical pharmacist.                                  |
+-----------------------------------------------------------------------------------------+
```

---

## 6. Mobile Experience & Responsive Layout

* **Adaptive Breakpoints**:
  * `Mobile (< 768px)`: Stacked single-column layouts with a persistent bottom tab bar (`Home`, `Prescriptions`, `Medicines`, `AI Chat`, `Profile`).
  * `Tablet (768px - 1024px)`: Collapsible drawer for prescription image inspection.
  * `Desktop (> 1024px)`: High-fidelity side-by-side synchronized viewports.
* **Touch Target Sizing**: Minimum $48 \times 48\text{px}$ touch targets for all action buttons and dropdown items.
