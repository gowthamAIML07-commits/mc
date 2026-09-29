# Retrieval-Augmented Generation (RAG) & Knowledge Architecture

## 1. Grounding Principles & Source Hierarchy

Medical information requires strict factual grounding. The RAG architecture operates under the following **uncompromising data principles**:
1. **Authoritative Sources Only**: No unverified blog posts, forum discussions, or random web crawls are ever ingested.
2. **Deterministic Precedence**: Direct structural facts (contraindications, drug-drug interactions, maximum daily doses) are retrieved from verified database tables (RxNorm / Structured Interaction DB) before querying dense vector indexes.
3. **Traceable Citations**: Every generated medical claim must anchor to an explicit, readable source ID and paragraph reference.

### Authoritative Document Repositories
* **FDA Structured Product Labeling (DailyMed / OpenFDA)**: Comprehensive indications, black-box warnings, contraindications, adverse reactions, and pediatric/geriatric use guidelines.
* **National Library of Medicine (NLM / MedlinePlus)**: Patient-friendly drug summaries, storage advice, missed-dose protocols.
* **PubMed Central (PMC Open Access Subset)**: Peer-reviewed clinical pharmacology and clinical trial evidence (CC BY / Open Access only).
* **Indian Pharmacopoeia Commission (IPC) & CDSCO**: Approved formulations, schedules, and regulatory guidance for Indian clinical contexts.

---

## 2. Ingestion & Chunking Pipeline

```
+-------------------------------------------------------------------------------+
|                             RAW SOURCE DOCUMENTS                              |
|           (DailyMed XML/JSON, MedlinePlus HTML, PMC Open Access XML)          |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                       CLEANING & SECTION EXTRACTION                           |
|  - Strip boilerplate, navigation, markup                                      |
|  - Parse standardized clinical sections:                                      |
|    [INDICATIONS, DOSAGE, CONTRAINDICATIONS, WARNINGS, INTERACTIONS, ADVERSE]  |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                      SEMANTIC MEDICAL CHUNKING                                |
|  - Section-aware boundaries (never split across clinical warning boundaries)  |
|  - Target chunk size: 384-512 tokens with 64-token overlap                    |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                      METADATA ENRICHMENT & TAXONOMY                           |
|  - RxCUI, Generic Name, Brand Names, Section Type, Audience, License, Date    |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                      DENSE & SPARSE EMBEDDING GENERATION                      |
|  - Dense: BAAI/bge-m3 (1024 dimensions, multilingual, clinical vocabulary)    |
|  - Sparse: BM25 / SPLADE lexical representation for exact drug names          |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                            QDRANT VECTOR DATABASE                             |
|  - Collections: medical_documents, drug_information, literature, faq         |
+-------------------------------------------------------------------------------+
```

---

## 3. Qdrant Collection Schema & Metadata Taxonomy

All vectors are indexed with dense embeddings (`dim: 1024`, cosine metric) and payload indexes for rapid metadata filtering:

```json
{
  "id": "doc_amox_500_warnings_004",
  "vector": [0.0124, -0.0452, 0.0891, "..."],
  "payload": {
    "source": "FDA_DailyMed",
    "source_id": "SPL-78491-AMOX",
    "title": "Amoxicillin Capsule - Prescribing Information",
    "section": "WARNINGS_AND_PRECAUTIONS",
    "section_title": "Severe Cutaneous Adverse Reactions (SCAR)",
    "date": "2023-11-01",
    "medicine": "Amoxicillin",
    "rxnorm_cui": "308189",
    "drug_classes": ["Penicillin Antibacterial", "Beta-lactam"],
    "audience": "clinical",
    "license": "Public Domain (US Gov)",
    "document_type": "official_drug_label",
    "content": "Severe cutaneous adverse reactions (SCAR), such as Stevens-Johnson syndrome (SJS), toxic epidermal necrolysis (TEN), drug reaction with eosinophilia and systemic symptoms (DRESS), and acute generalized exanthematous pustulosis (AGEP) have been reported in patients receiving amoxicillin..."
  }
}
```

### Dedicated Collections
1. **`drug_information`**: Official FDA/DailyMed drug package inserts and CDSCO product monographs.
2. **`medical_documents`**: Clinical practice guidelines, consensus statements, and hospital protocols.
3. **`clinical_literature`**: PMC open-access pharmacology research articles and pharmacokinetic studies.
4. **`faq`**: Curated patient-oriented Q&As (MedlinePlus format).

---

## 4. Hybrid Retrieval & Re-ranking Architecture

```
User Query: "Can an asthmatic take Ibuprofen with Aspirin?"
                           |
            +--------------+---------------+
            |                              |
            v                              v
[Dense Vector Retrieval]       [Sparse BM25 / Lexical Filter]
(BGE-M3 in Qdrant, k=25)       (Exact match: "Ibuprofen", "Aspirin", "Asthma")
            |                              |
            +--------------+---------------+
                           |
                           v
           [Reciprocal Rank Fusion (RRF)]
           Merged Top-30 Unique Chunks
                           |
                           v
         [Cross-Encoder Reranker: BGE-Reranker-Large]
         Compute Query-Passage Relevance Score S_rel
                           |
                           v
           [Top-5 Filtered Chunks (S_rel > 0.65)]
                           |
                           v
         [Context Construction + Safety Guardrails]
                           |
                           v
            [Open LLM (Llama-3-8B / Mistral-7B)]
```

---

## 5. Medical Response Construction & Prompt Strategy

The system prompt enforces strict clinical constraints, preventing speculation and requiring inline source citations.

### System Prompt Template
```
You are the AI Medicine Assistant. You provide objective, evidence-based medical information based solely on the verified context provided below.

RULES:
1. Ground every medical statement strictly in the provided Context.
2. If the context does not contain sufficient information, explicitly state: "I do not have verified clinical evidence in my current database to answer this question."
3. Cite your sources using bracketed references [Source X: Document Title - Section].
4. Include both a "Key Summary" in simple language and a "Clinical Details" section where appropriate.
5. NEVER recommend altering a prescription dosage or stopping a medication without consulting the prescribing physician.
6. Append the standard clinical disclaimer at the conclusion.

CONTEXT:
{retrieved_context_chunks}

USER QUERY:
{sanitized_user_query}
```

### Response Structure Format
1. **Direct Answer**: Concise 1-2 sentence core answer.
2. **Important Clinical Information**: Uses, mechanism, or instructions.
3. **Precautions & Safety Warnings**: Common side effects, contraindicated conditions.
4. **Verified Sources**: Expandable citations with document IDs.
5. **Medical Disclaimer**: Standard regulatory disclaimer.

---

## 6. Deterministic Drug Interaction Engine

Drug interaction queries **do not rely on generative hallucination**. The pipeline intercepts interaction queries through a deterministic matrix:

```
[User checks: "Metformin + Glipizide + Ramipril"]
                       |
        [Generate Pairwise Combinations]
   1. (Metformin, Glipizide)
   2. (Metformin, Ramipril)
   3. (Glipizide, Ramipril)
                       |
        [PostgreSQL Drug Interaction Table]
   SELECT * FROM drug_interactions 
   WHERE (drug_a_cui = :cui1 AND drug_b_cui = :cui2)
      OR (drug_a_cui = :cui2 AND drug_b_cui = :cui1);
                       |
        +--------------+--------------+
        |                             |
[Interaction Found]         [No Record Found]
Severity: MODERATE          "Unable to verify this interaction from available sources.
Description: Risk of severe  Consult your pharmacist or physician."
hypoglycemia when combined.
Source: FDA Package Insert
```
