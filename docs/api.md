# REST & Streaming API Specification (v1)

## 1. Overview & Protocol Conventions

The **AI Medicine Assistant API** is built on FastAPI and adheres to RESTful design patterns with OpenAPI 3.1 compliance. Streaming endpoints utilize Server-Sent Events (SSE).

### General Standards
* **Base URL**: `/api/v1`
* **Authentication**: Bearer JWT tokens in `Authorization: Bearer <token>` header.
* **Content Negotiation**: `application/json` for standard endpoints; `multipart/form-data` for file uploads; `text/event-stream` for chat streaming.
* **Standard Error Response**:
  ```json
  {
    "error": {
      "code": "ENTITY_NOT_FOUND",
      "message": "Prescription with ID 'rx_123' was not found.",
      "request_id": "req_872c6e3b-9a4f-4d92-bf39-4d6cb82e9871",
      "timestamp": "2024-03-15T10:30:00Z"
    }
  }
  ```

---

## 2. Authentication & User Management

### 2.1 Register User
* **Endpoint**: `POST /api/v1/auth/register`
* **Request Body**:
  ```json
  {
    "email": "doctor@hospital.org",
    "password": "SecurePassword123!",
    "full_name": "Dr. S. K. Mukherjee",
    "role": "clinician"
  }
  ```
* **Response (201 Created)**:
  ```json
  {
    "user_id": "usr_550e8400-e29b-41d4-a716-446655440000",
    "email": "doctor@hospital.org",
    "full_name": "Dr. S. K. Mukherjee",
    "role": "clinician",
    "created_at": "2024-03-15T10:00:00Z"
  }
  ```

### 2.2 User Login
* **Endpoint**: `POST /api/v1/auth/login`
* **Request Body**:
  ```json
  {
    "email": "doctor@hospital.org",
    "password": "SecurePassword123!"
  }
  ```
* **Response (200 OK)**:
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 86400,
    "user": {
      "id": "usr_550e8400-e29b-41d4-a716-446655440000",
      "email": "doctor@hospital.org",
      "full_name": "Dr. S. K. Mukherjee",
      "role": "clinician"
    }
  }
  ```

---

## 3. Prescription Processing Endpoints

### 3.1 Upload Prescription Image
* **Endpoint**: `POST /api/v1/prescriptions/upload`
* **Content-Type**: `multipart/form-data`
* **Form Parameters**:
  * `file`: Binary image file (`image/png`, `image/jpeg`, `image/webp`, `application/pdf`).
  * `language_hint`: Optional (`en`, `hi`, `auto`).
* **Response (202 Accepted / 200 OK)**:
  ```json
  {
    "prescription_id": "rx_987fbc82_a12e_489b",
    "status": "PROCESSED",
    "image_quality": {
      "blur_score": 78.4,
      "is_acceptable": true,
      "skew_angle": 1.2
    },
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
        "id": "med_item_001",
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
        "verification_status": "verified",
        "bounding_box": {"x": 120, "y": 340, "width": 480, "height": 65}
      }
    ],
    "instructions": ["Take after meals with water"],
    "warnings": [],
    "overall_confidence": 0.91,
    "review_required": false
  }
  ```

### 3.2 List User Prescriptions
* **Endpoint**: `GET /api/v1/prescriptions?page=1&limit=20&status=verified`
* **Response (200 OK)**:
  ```json
  {
    "items": [
      {
        "prescription_id": "rx_987fbc82_a12e_489b",
        "doctor_name": "Dr. S. K. Mukherjee",
        "date": "2024-03-15",
        "medicine_count": 2,
        "overall_confidence": 0.91,
        "status": "verified",
        "created_at": "2024-03-15T10:15:00Z"
      }
    ],
    "total": 1,
    "page": 1,
    "limit": 20,
    "total_pages": 1
  }
  ```

### 3.3 Get Prescription Details
* **Endpoint**: `GET /api/v1/prescriptions/{id}`
* **Response (200 OK)**: Full Prescription JSON (Schema matching Section 3.1).

### 3.4 Verify & Update Prescription (Human Review)
* **Endpoint**: `POST /api/v1/prescriptions/{id}/verify`
* **Request Body**:
  ```json
  {
    "patient": {
      "name": "Aarav Sharma",
      "age": 34,
      "gender": "Male"
    },
    "doctor": {
      "name": "Dr. S. K. Mukherjee",
      "registration": "WBMC-48291"
    },
    "medicines": [
      {
        "id": "med_item_001",
        "normalized_name": "Amoxicillin 500 MG Oral Tablet",
        "rxnorm_id": "308189",
        "strength": "500 mg",
        "dose": "1 tablet",
        "frequency": "3 times daily",
        "duration": "5 days",
        "user_confirmed": true
      }
    ],
    "user_notes": "Corrected patient age from 31 to 34"
  }
  ```
* **Response (200 OK)**: Updated prescription object with status `"verified"`.

---

## 4. Medicine Information & Search Endpoints

### 4.1 Search Medicines (Autocomplete & Fuzzy)
* **Endpoint**: `POST /api/v1/medicines/search`
* **Request Body**:
  ```json
  {
    "query": "Amox",
    "limit": 5,
    "include_brands": true
  }
  ```
* **Response (200 OK)**:
  ```json
  {
    "results": [
      {
        "rxnorm_id": "308189",
        "name": "Amoxicillin 500 MG Oral Tablet",
        "generic_name": "Amoxicillin",
        "brand_names": ["Amoxil", "Moxatag"],
        "drug_class": "Penicillin Antibacterial",
        "match_type": "prefix"
      },
      {
        "rxnorm_id": "312320",
        "name": "Amoxicillin 250 MG Oral Capsule",
        "generic_name": "Amoxicillin",
        "brand_names": ["Amoxil"],
        "drug_class": "Penicillin Antibacterial",
        "match_type": "prefix"
      }
    ]
  }
  ```

### 4.2 Get Medicine Monograph
* **Endpoint**: `GET /api/v1/medicines/{id}` (where `{id}` is RxCUI or Slug)
* **Response (200 OK)**:
  ```json
  {
    "rxnorm_id": "308189",
    "name": "Amoxicillin 500 MG Oral Tablet",
    "generic_name": "Amoxicillin",
    "brand_names": ["Amoxil", "Moxatag", "Augmentin (combination)"],
    "active_ingredient": "Amoxicillin",
    "drug_class": "Penicillin Beta-lactam Antibiotic",
    "uses": [
      "Treatment of infections of the ear, nose, and throat",
      "Lower respiratory tract infections",
      "Skin and skin structure infections"
    ],
    "common_side_effects": [
      "Nausea",
      "Diarrhea",
      "Vomiting",
      "Mild skin rash"
    ],
    "warnings": [
      "Anaphylaxis and serious hypersensitivity reactions reported",
      "Clostridioides difficile-associated diarrhea (CDAD)"
    ],
    "contraindications": [
      "History of serious hypersensitivity reaction to amoxicillin or other beta-lactams"
    ],
    "source": "FDA DailyMed / NLM RxNorm",
    "last_updated": "2024-01-15"
  }
  ```

---

## 5. Drug Interaction Checking Endpoint

### 5.1 Check Multi-Drug Interactions
* **Endpoint**: `POST /api/v1/interactions/check`
* **Request Body**:
  ```json
  {
    "medicines": [
      {"name": "Amoxicillin", "rxnorm_id": "308189"},
      {"name": "Methotrexate", "rxnorm_id": "6851"},
      {"name": "Paracetamol", "rxnorm_id": "161"}
    ]
  }
  ```
* **Response (200 OK)**:
  ```json
  {
    "analyzed_count": 3,
    "pair_combinations_checked": 3,
    "interactions": [
      {
        "drug_a": "Amoxicillin",
        "drug_b": "Methotrexate",
        "severity": "HIGH",
        "description": "Concomitant use of penicillins can reduce renal clearance of methotrexate, resulting in increased serum levels and risk of toxicity.",
        "management": "Monitor serum methotrexate levels closely and adjust dose as recommended by physician.",
        "source": "FDA Prescribing Information #SPL-88210"
      }
    ],
    "disclaimer": "Absence of a recorded interaction does not imply clinical safety. Always consult a physician."
  }
  ```

---

## 6. AI Conversational Chat Endpoints

### 6.1 Send Chat Message (Streaming / Non-Streaming)
* **Endpoint**: `POST /api/v1/chat`
* **Request Body**:
  ```json
  {
    "session_id": "sess_4982a17b-03fc-41b9-a2ce-623b3a628172",
    "message": "What should I do if I missed a dose of Amoxicillin 500mg?",
    "stream": false,
    "technical_level": "simple"
  }
  ```
* **Response (200 OK)**:
  ```json
  {
    "session_id": "sess_4982a17b-03fc-41b9-a2ce-623b3a628172",
    "message_id": "msg_09182374",
    "response": "If you miss a dose of Amoxicillin, take it as soon as you remember. However, if it is almost time for your next scheduled dose, skip the missed dose and continue with your regular schedule. Do not double up on doses.",
    "intent": "MISSED_DOSE",
    "safety_risk": "LOW",
    "sources": [
      {
        "title": "MedlinePlus: Amoxicillin Patient Guide",
        "source_id": "NLM-MEDP-1029",
        "section": "Missed Dose Instructions",
        "url": "https://medlineplus.gov/druginfo/meds/a685001.html"
      }
    ],
    "disclaimer": "This information is educational and does not replace your doctor's instructions."
  }
  ```

### 6.2 Get Chat Sessions
* **Endpoint**: `GET /api/v1/chat/sessions`
* **Response (200 OK)**: List of user conversation sessions with message previews and timestamps.

### 6.3 Get Chat Session History
* **Endpoint**: `GET /api/v1/chat/sessions/{id}`
* **Response (200 OK)**: Full message history for specified session.

---

## 7. Quality Feedback & Health Endpoints

### 7.1 Submit User Feedback
* **Endpoint**: `POST /api/v1/feedback`
* **Request Body**:
  ```json
  {
    "reference_type": "prescription_ocr",
    "reference_id": "rx_987fbc82_a12e_489b",
    "rating": 5,
    "comment": "Accurately transcribed difficult handwritten dosage",
    "correction_payload": null
  }
  ```
* **Response (201 Created)**: `{"status": "feedback_recorded"}`

### 7.2 System Health Probe
* **Endpoint**: `GET /api/v1/health`
* **Response (200 OK)**:
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "database": "connected",
    "redis": "connected",
    "qdrant": "connected",
    "gpu_available": true,
    "model_device": "cuda:0",
    "active_llm": "meta-llama/Meta-Llama-3-8B-Instruct"
  }
  ```
