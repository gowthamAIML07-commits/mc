# AI Medicine Assistant
### Prescription OCR • Handwritten Medicine Recognition • Knowledge Graph Normalization • Deterministic Interaction Checker • Clinical RAG Assistant

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js: 14](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red.svg)](https://qdrant.tech/)

---

> **CRITICAL MEDICAL DISCLAIMER**  
> **The AI Medicine Assistant is an educational and informational tool.** It does not provide medical diagnosis, prescribe pharmaceutical treatment, or replace a qualified healthcare professional. For medical emergencies, immediately contact your local emergency response service (e.g., **112 / 108** in India, **911** in the US).

---

## 1. Project Overview

The **AI Medicine Assistant** is an end-to-end, open-source medical informatics system designed to digitize, verify, and explain medical prescriptions. It solves the critical challenge of illegible handwritten prescriptions, dosage confusion, and drug-drug interactions through a multi-stage, explainable AI pipeline grounded in standardized medical ontologies (**RxNorm** & **RxTerms**).

### Key Features
1. **Prescription Image Preprocessing & OCR**: Deskew, denoise, and extract full-page printed and handwritten prescriptions.
2. **Handwritten Medicine Crop Recognition**: Specialized handwriting recognition for complex doctor handwriting.
3. **Ontology Normalization & Verification**: Deterministic mapping of noisy trade names to standardized RxNorm RxCUIs and RxTerms codes.
4. **Human-in-the-Loop Review UI**: Side-by-side visual bounding-box inspector for user confirmation and correction.
5. **Deterministic Drug Interaction Checker**: 100% verified pairwise interaction lookup (zero generative hallucination).
6. **Clinical RAG Assistant**: Hybrid dense/sparse vector retrieval (Qdrant + BGE-M3 + BGE-Reranker) grounded in authoritative FDA drug monographs.
7. **Isolated Safety Firewall**: Instant rule-based detection and escalation for emergencies, overdoses, and high-risk queries.
8. **Indian Prescription Workflow Support**: Optimized for Indian clinical formats, regional abbreviations, and CDSCO drug schedules.

---

## 2. System Architecture

```
                    +-----------------------+
                    |      User / Client    |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    |    Next.js 14 UI      |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    |   FastAPI Backend     |
                    +---+---------------+---+
                        |               |
        +---------------+               +---------------+
        |                                               |
        v                                               v
+-------------------------------+               +-------------------------------+
| Prescription Vision Pipeline  |               | RAG & Knowledge Engine        |
| - Preprocessing (OpenCV)      |               | - Safety Firewall (Rules)     |
| - Document OCR (Paddle/TrOCR) |               | - Intent & NER (BioBERT)      |
| - Handwriting Model (CRNN)    |               | - Vector Search (Qdrant)      |
| - Normalizer (RxNorm/RxTerms) |               | - Reranker (BGE-Reranker)     |
| - Human Review Stage          |               | - Open LLM (Llama-3/Mistral)  |
+---------------+---------------+               +---------------+---------------+
                |                                               |
                +-----------------------+-----------------------+
                                        |
                                        v
                    +-----------------------------------+
                    |    Data & Knowledge Layer         |
                    |  PostgreSQL 16 | Qdrant | Redis 7 |
                    +-----------------------------------+
```

---

## 3. Multi-Task ML Datasets & Licensing Matrix

| Dataset | Role in Pipeline | License | Source / Status |
| :--- | :--- | :--- | :--- |
| **Indian Medical Prescription OCR** | Full-page OCR & field extraction | CC BY 4.0 | Open Access Clinical Research |
| **Doctor Handwritten Prescription BD** | Handwritten medicine word recognition | CC BY 4.0 | Kaggle / Mendeley (~4,680 samples) |
| **RxNorm** | Drug ontology & verification knowledge base | UMLS License | NLM / NIH (Knowledge base) |
| **RxTerms** | Fast search indexing & autocomplete | Open Access | NLM / Public Domain |
| **FUNSD** | Layout analysis & key-value parsing pretraining | Academic Open | ICDAR / Non-commercial |
| **Synthetic Prescriptions** | Cold-start bootstrapping & augmentation | MIT | Internal Generator (~1,000 samples) |

*Full licensing details and governance rules are documented in [`docs/datasets.md`](docs/datasets.md) and [`data/registry/datasets.yaml`](data/registry/datasets.yaml).*

---

## 4. Documentation Index

* 📐 [**System Architecture**](docs/architecture.md): Deep-dive architectural design and component boundaries.
* 📊 [**Dataset Registry & Licenses**](docs/datasets.md): Licensing compliance and task segregation rules.
* 🤖 [**Machine Learning Pipeline**](docs/ml_pipeline.md): Vision models, handwriting recognition, and confidence scoring.
* 🔍 [**RAG Knowledge Base**](docs/rag.md): Hybrid vector retrieval, reranking, and deterministic drug interaction engine.
* 🛡️ [**Safety & Emergency Protocols**](docs/safety.md): Deterministic safety firewall and emergency triage matrices.
* 🎨 [**UI / UX Design System**](docs/ui-ux.md): Human-in-the-loop review interface and accessibility specs.
* 🔌 [**REST & Streaming API**](docs/api.md): Complete OpenAPI endpoint documentation.
* 🚀 [**Deployment & Operations**](docs/deployment.md): Docker, GPU/CPU sizing, and observability setup.

---

## 5. Quickstart & Local Setup

### Prerequisites
* Docker & Docker Compose (v2.20+)
* Python 3.11+
* Node.js 20 LTS & npm
* (Optional) NVIDIA GPU with CUDA 12.1+ for local neural model acceleration.

### Quick Start with Docker
```bash
# 1. Clone repository
git clone https://github.com/organization/medicine-ai.git
cd medicine-ai

# 2. Copy environment configuration
cp .env.example .env

# 3. Start platform services (Postgres, Redis, Qdrant, Backend, Frontend)
docker compose up -d
```

### Accessing Interfaces
* **Frontend Web Application**: [http://localhost:3000](http://localhost:3000)
* **Backend REST API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Qdrant Vector Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
* **MLflow Tracking Server**: [http://localhost:5000](http://localhost:5000)

---

## 6. Implementation Roadmap

The project executes across 17 structured engineering phases:
- [x] **Phase 1**: Repository Architecture & Design Specifications
- [ ] **Phase 2**: Dataset Registry, Manifest Validation & Seed Data
- [ ] **Phase 3**: Dataset Download Scripts & Checksum Verification
- [ ] **Phase 4**: Image Preprocessing & Sauvala/CLAHE Pipeline
- [ ] **Phase 5**: Full Prescription OCR & Layout Detection Engine
- [ ] **Phase 6**: Handwritten Medicine Recognition & CTC Decoding
- [ ] **Phase 7**: RxNorm & RxTerms Database Ingestion & Fuzzy Matcher
- [ ] **Phase 8**: Relational PostgreSQL Medical Database & Schemas
- [ ] **Phase 9**: Qdrant Vector DB Ingestion & Hybrid RAG Retrieval
- [ ] **Phase 10**: Open LLM Integration (Llama-3 / Mistral via `MODEL_NAME`)
- [ ] **Phase 11**: Deterministic Safety Layer & Emergency Triage
- [ ] **Phase 12**: FastAPI Production Endpoints & Rate Limiting
- [ ] **Phase 13**: Next.js 14 Frontend & Side-by-Side Human Review UI
- [ ] **Phase 14**: Comprehensive Model & System Evaluation
- [ ] **Phase 15**: Security Hardening, Audit Logs & Privacy Controls
- [ ] **Phase 16**: Docker Multi-Stage Optimization & GPU Runtime
- [ ] **Phase 17**: End-to-End Testing & Production Verification
