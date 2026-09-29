# Deployment, Infrastructure & Production Operations Specification

## 1. Production Architecture Topology

The **AI Medicine Assistant** infrastructure is designed for high availability, zero-downtime rolling upgrades, containerized portability, and hybrid GPU/CPU orchestration.

```
                              [Internet / Client Requests]
                                            |
                                            v
                             +------------------------------+
                             |   NGINX / Cloudflare Proxy   |
                             |   SSL, WAF, Rate Limiting    |
                             +--------------+---------------+
                                            |
                    +-----------------------+-----------------------+
                    |                                               |
                    v                                               v
     +------------------------------+                +------------------------------+
     |   Next.js Frontend Cluster   |                |   FastAPI Backend Cluster    |
     |   (Node.js 20 LTS / SSR)     |                |   (Python 3.11+ / Uvicorn)   |
     +------------------------------+                +--------------+---------------+
                                                                    |
             +-----------------------+-----------------------+------+-----------------------+
             |                       |                       |                              |
             v                       v                       v                              v
+------------------------+ +-------------------+ +-----------------------+ +------------------------+
|  PostgreSQL 16 Cluster | |  Redis 7 Cluster  | |  Qdrant Vector DB     | |  AI Inference Worker   |
|  (ACID Relational DB)  | |  (Cache & Queue)  | |  (Dense/Sparse Index) | |  (GPU / PyTorch / LLM) |
+------------------------+ +-------------------+ +-----------------------+ +------------------------+
```

---

## 2. Containerized Service Specifications

The application uses multi-stage Docker builds to ensure lean production images:

| Service Container | Base Image | Purpose | Port | Resource Limits (Prod) |
| :--- | :--- | :--- | :--- | :--- |
| **`web`** | `node:20-alpine` | Next.js 14 Frontend UI | 3000 | 2 CPU, 2 GB RAM |
| **`api`** | `python:3.11-slim` | FastAPI Gateway & Business Logic | 8000 | 4 CPU, 4 GB RAM |
| **`worker-ai`** | `nvidia/cuda:12.1.1-runtime-ubuntu22.04` | Vision, OCR, and Transformer Models | 8001 | 8 CPU, 16 GB RAM + 1 GPU (16GB VRAM) |
| **`db`** | `postgres:16-alpine` | Relational Storage & RxNorm | 5432 | 4 CPU, 8 GB RAM |
| **`vector-db`** | `qdrant/qdrant:latest` | Vector Index (BGE-M3 Embeddings) | 6333 | 4 CPU, 8 GB RAM |
| **`cache`** | `redis:7-alpine` | Token Cache & Rate Limiting | 6379 | 1 CPU, 2 GB RAM |
| **`mlflow`** | `python:3.11-slim` | ML Experiment Tracking | 5000 | 1 CPU, 2 GB RAM |

---

## 3. Environment Configuration & Profiles

The system configures runtime behavior using hierarchical environment variables (`.env`):

### Key Environment Variables
* `APP_ENV`: `development` | `staging` | `production`
* `DEVICE`: `auto` | `cuda` | `cpu`
* `MODEL_NAME`: Hugging Face model identifier (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`)
* `EMBEDDING_MODEL`: `BAAI/bge-m3`
* `RERANKER_MODEL`: `BAAI/bge-reranker-large`
* `DATABASE_URL`: `postgresql+asyncpg://user:pass@db:5432/medicine_ai`
* `REDIS_URL`: `redis://cache:6379/0`
* `QDRANT_URL`: `http://vector-db:6333`
* `JWT_SECRET_KEY`: High-entropy 256-bit secret key

---

## 4. Hardware Sizing & Scaling Guidelines

### 4.1 Development Profile (CPU Fallback)
* **Processor**: 4-Core x86_64 CPU (Intel i5/i7 or AMD Ryzen) or Apple Silicon (M1/M2/M3).
* **RAM**: 16 GB minimum.
* **Storage**: 20 GB SSD.
* **Execution**: Employs INT8 quantized ONNX OCR and quantized GGUF LLMs (via llama.cpp / Ollama) or remote model endpoints.

### 4.2 Production Profile (Full Acceleration)
* **Processor**: 16+ vCPU.
* **RAM**: 64 GB ECC RAM.
* **GPU**: 1x NVIDIA A10G (24GB VRAM) or 1x NVIDIA L4 (24GB VRAM) or A100 (40GB/80GB).
* **Storage**: 500 GB NVMe SSD.
* **Throughput**: ~45 concurrent prescription OCR extractions/min + ~120 RAG chat tokens/sec.

---

## 5. Observability, Structured Logging & Health Monitoring

### 5.1 Structured Logging (JSON)
All logs are emitted in structured JSON with mandatory trace and correlation IDs:
```json
{
  "timestamp": "2024-03-15T10:30:15.210Z",
  "level": "INFO",
  "logger": "medicine_ai.ocr.pipeline",
  "request_id": "req_872c6e3b-9a4f-4d92",
  "action": "prescription_ocr_complete",
  "duration_ms": 421.5,
  "confidence": 0.92,
  "verification_status": "verified"
}
```

### 5.2 Prometheus Metrics
Exported at `/metrics` for Prometheus scraping:
* `http_requests_total{endpoint, status_code}`
* `ocr_processing_duration_seconds{status}`
* `rag_retrieval_duration_seconds{collection}`
* `llm_token_generation_duration_seconds`
* `safety_emergency_events_total{category}`

### 5.3 Health Check Probes
* `GET /api/v1/health`: Returns composite health of Database, Redis, Qdrant, Model Weights, and GPU status.
