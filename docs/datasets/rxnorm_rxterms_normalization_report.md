# RxNorm / RxTerms Clinical Lexical Normalization Audit Report

**Resource**: RxNorm (NLM) + RxTerms Terminology Index  
**Task**: Clinical Entity Normalization & Drug Verification (Pipeline E)  
**Authorities**: National Library of Medicine (NIH)  
**License**: UMLS Metathesaurus Agreement / Open Access RxTerms  
**Knowledge Base Hash**: `074909cc643919ed32b589cd341c1d67aa8f3f3d3747a17ae2d20b204791a236`  
**Manifest Path**: [`data/manifests/rxnorm_rxterms_manifest.json`](file:///c:/Users/AIML/Documents/clg%20mc/data/manifests/rxnorm_rxterms_manifest.json)  
**Vocabulary Path**: [`data/processed/normalization/approved_vocabulary.json`](file:///c:/Users/AIML/Documents/clg%20mc/data/processed/normalization/approved_vocabulary.json)

---

## 1. Resource Overview & Architecture

The clinical lexical normalization engine provides deterministic, 4-tier mapping from noisy OCR medicine text (including doctor handwriting, trade names, phonetic misspellings, and clinical abbreviations) into standard **RxNorm Concept Unique Identifiers (RxCUIs)** and **RxTerms** prescription descriptors.

```mermaid
flowchart TD
    Raw["Raw OCR String (e.g. 'Tab. Dolo 650')"] --> Abbrev["Tier 0: Abbreviation & Dosage Stripping"]
    Abbrev --> ExactTier 1: Exact / Alias Match?
    Exact -- Yes --> Verified["Verified RxCUI (161) - Conf: 0.98"]
    Exact -- No --> PhoneticTier 2: Phonetic Key Match (Metaphone/Soundex)?
    Phonetic -- Yes --> PhoneCand["Candidate Filter + Levenshtein Ranking"]
    PhoneCand --> VerifiedPhone["Verified RxCUI - Conf: 0.85-0.95"]
    Phonetic -- No --> Fuzzy["Tier 3: Global Fuzzy Search (Levenshtein)"]
    Fuzzy --> ThresholdSimilarity >= 0.70?
    Threshold -- Yes --> Candidate["Fuzzy Matched Concept - Conf: 0.70-0.85"]
    Threshold -- No --> Unverified["Unverified / Review Required"]
```

---

## 2. Indexed Concepts & Formulations

| RxCUI | Standard Clinical Drug Name | Active Ingredient | Strength & Form | ATC Code | Trade Names / Aliases |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **308189** | Amoxicillin 500 MG Oral Tablet | Amoxicillin | 500 mg Tablet | J01CA04 | Amoxil, Moxatag, Novamox, Moxikind |
| **161** | Acetaminophen 650 MG Oral Tablet | Paracetamol | 650 mg Tablet | N02BE01 | Dolo 650, Crocin, Calpol, Panadol, Tylenol |
| **198440** | Azithromycin 500 MG Oral Tablet | Azithromycin | 500 mg Tablet | J01FA10 | Zithromax, Azithral, Azee, Zady |
| **860975** | Metformin HCl 500 MG Oral Tablet | Metformin | 500 mg Tablet | A10BA02 | Glucophage, Glycomet, Glyciphage |
| **310489** | Cetirizine HCl 10 MG Oral Tablet | Cetirizine | 10 mg Tablet | R06AE07 | Zyrtec, Cetzine, Alerid, Okacet |
| **312615** | Pantoprazole 40 MG DR Tablet | Pantoprazole | 40 mg DR Tab | A02BC02 | Protonix, Pan 40, Pantocid, Pantonix |
| **310965** | Ibuprofen 400 MG Oral Tablet | Ibuprofen | 400 mg Tablet | M01AE01 | Advil, Motrin, Brufen, Ibugesic |
| **311689** | Montelukast 10 MG Oral Tablet | Montelukast | 10 mg Tablet | R03DC03 | Singulair, Montair, Montek, Monticope |
| **312961** | Omeprazole 20 MG DR Capsule | Omeprazole | 20 mg DR Cap | A02BC01 | Prilosec, Omez, Omep, Ocid |
| **310385** | Ciprofloxacin 500 MG Oral Tablet | Ciprofloxacin | 500 mg Tablet | J01MA02 | Cipro, Ciplox, Cifran, Ciprobid |
| **314076** | Amlodipine 5 MG Oral Tablet | Amlodipine | 5 mg Tablet | C08CA01 | Norvasc, Amlong, Amtas, Amlip |
| **310798** | Atorvastatin 10 MG Oral Tablet | Atorvastatin | 10 mg Tablet | C10AA05 | Lipitor, Atorva, Storvas, Tonact |
| **311354** | Levocetirizine 5 MG Oral Tablet | Levocetirizine | 5 mg Tablet | R06AE09 | Xyzal, Levocet, Lcz, 1-Al |
| **314164** | Telmisartan 40 MG Oral Tablet | Telmisartan | 40 mg Tablet | C09CA07 | Micardis, Telma, Telmikind, Cresar |
| **312984** | Ranitidine 150 MG Oral Tablet | Ranitidine | 150 mg Tablet | A02BA02 | Zantac, Rantac, Aciloc, Histac |

---

## 3. Clinical Abbreviation Dictionary

The normalizer includes structured resolution for medical abbreviations across 3 domains:
1. **Dosage Forms** (`tab`, `cap`, `syr`, `inj`, `susp`, `oint`, `soln`, `drp`).
2. **Frequency Schedules** (`OD`, `BD`/`BID`, `TDS`/`TID`, `QID`/`QDS`, `HS`, `SOS`, `STAT`, `PRN`, `AC`, `PC`).
3. **Common Drug Shorthands** (`PCM` -> Paracetamol, `AMX` -> Amoxicillin, `AZM` -> Azithromycin, `MET` -> Metformin, `PANTO` -> Pantoprazole, `HCQ` -> Hydroxychloroquine).

---

## 4. Compliance & License Safeguards

- Raw UMLS Metathesaurus archive downloads are restricted behind NIH UTS credentials and are not redistributed in git.
- The project distributes only clean, processed JSON lookup tables derived with full attribution to NLM RxNorm and RxTerms.
- Synthetic generators and RAG retrievers reference the canonical vocabulary at `data/processed/normalization/approved_vocabulary.json`.
