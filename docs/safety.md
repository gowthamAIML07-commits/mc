# Clinical Safety Architecture, Risk Stratification & Emergency Protocols

## 1. Safety-First Medical Philosophy

The **AI Medicine Assistant** operates under a zero-compromise safety paradigm. In clinical computing, probabilistic language models are susceptible to hallucinations, sycophancy, and uncalibrated confidence. Therefore, the safety architecture is designed as an **independent, deterministic firewall** executed before and after any interaction with LLMs.

### Core Safety Tenets
1. **Separation of Concerns**: Safety classification and emergency interception do not rely on LLM prompts; they are hardcoded deterministic and dedicated classifier guardrails.
2. **Never Diagnose**: The system never provides clinical diagnoses (e.g., "You are having a myocardial infarction").
3. **Never Prescribe or Modify Doses**: The system will never advise a user to change, increase, or discontinue prescription dosages.
4. **Immediate Emergency Escalation**: Queries exhibiting critical red-flag symptoms immediately bypass conversational generation and trigger clear, prominent emergency directives.

---

## 2. Five-Tier Risk Classification Matrix

```
                                  USER QUERY / INPUT
                                          |
                                          v
                   +-----------------------------------------------+
                   |      PRE-EXECUTION SAFETY FIREWALL (MODEL F)  |
                   |  (Deterministic Regex Matrix + BioBERT Triage)|
                   +----------------------+------------------------+
                                          |
        +------------------+--------------+---------------+------------------+
        |                  |                              |                  |
        v                  v                              v                  v
  [TIER 5: EMERGENCY] [TIER 4: HIGH RISK]          [TIER 3: MODERATE] [TIER 1 & 2: LOW/OOS]
        |                  |                              |                  |
        v                  v                              v                  v
  +---------------+  +--------------------------+   +--------------+   +---------------+
  | Instant Red   |  | Urgent Clinical Advisory |   | Standard     |   | Informational |
  | Escalation    |  | + Emergency Hotlines     |   | Grounded RAG |   | RAG Workflow  |
  | Banner (No LLM|  | + Contextual Disclaimer  |   | + Cautions   |   | (Full Ground) |
  +---------------+  +--------------------------+   +--------------+   +---------------+
```

| Risk Level | Definition & Clinical Triggers | System Action | Response Payload |
| :--- | :--- | :--- | :--- |
| **`EMERGENCY`** | Immediate life-threatening scenarios: acute chest pain radiating to arm/jaw, severe shortness of breath, anaphylaxis (throat closing, lip swelling), sudden loss of consciousness, active seizure, massive hemorrhage, acute poisoning, overdose of lethal quantity, or explicit self-harm intent with pharmaceuticals. | **Complete LLM Bypass**: Halts generative pipeline immediately. Logs high-priority safety event. | Prominent emergency banner with regional emergency numbers (112 / 108 / 911 / Poison Control), basic first-aid position advice, and instructions to seek immediate medical attention. |
| **`HIGH`** | Severe potential adverse drug events: high-risk medication combinations (e.g. Warfarin + NSAIDs, Methotrexate overdosing), pediatric ingestion of adult drugs, pregnancy category X drug inquiries with suspected pregnancy. | **Cautious Structured Response**: Displays verified clinical warning card, blocks speculative text, mandates consultation with the prescribing physician. | Bold warning box detailing documented contraindication and explicit instruction: "Do not take this combination without direct physician approval." |
| **`MODERATE`** | Common adverse effects, missed dose confusion, mild drug interactions, off-label usage questions. | **Standard Grounded RAG**: Fetches verified drug label sections (Warnings / Dosage). | Clear, grounded explanation with source citations and standard clinical advisory. |
| **`LOW`** | General informational queries: storage conditions, pill shape/color identification, pharmaceutical manufacturer, generic equivalents. | **Full Grounded RAG**: Answers directly from verified knowledge base. | Complete informational response with structured summary and citations. |
| **`OUT_OF_SCOPE`** | Non-medical queries (coding, finance, sports) or hazardous non-therapeutic synthesis queries (explosives, illicit narcotics synthesis). | **Safe Refusal**: Politely declines non-medical or hazardous requests. | "This assistant is restricted to prescription understanding and evidence-based medicine information." |

---

## 3. Deterministic Emergency & High-Risk Rules Matrix

The deterministic engine evaluates queries against curated clinical regex patterns across 10 core safety domains:

```python
# Sample rule definitions from ml/safety/rules.py
EMERGENCY_TRIGGER_PATTERNS = {
    "anaphylaxis": [
        r"\b(throat closing|can'?t breathe|swelling (of )?(lips|tongue|throat)|difficulty breathing after taking)\b",
        r"\b(severe allergic reaction|anaphylact(ic|oid))\b"
    ],
    "cardiac_emergency": [
        r"\b(crushing chest pain|chest pressure (radiating|spreading)|pain in (left arm|jaw)|suspected heart attack)\b"
    ],
    "overdose_poisoning": [
        r"\b(took (whole bottle|entire strip|too many pills|excess dose)|overdosed on|swallowed poison|drank bleach)\b",
        r"\b(child swallowed (pills|medicine|capsules|tablets))\b"
    ],
    "neurological_crisis": [
        r"\b(unconscious|unresponsive|having a seizure|convulsing|sudden numbness on one side|slurred speech)\b"
    ],
    "severe_hemorrhage": [
        r"\b(vomiting blood|coughing up blood|severe bleeding won'?t stop|black tarry stools after)\b"
    ],
    "self_harm": [
        r"\b(want to end my life|how many pills to (die|kill myself)|lethal dose of|overdose intentionally)\b"
    ]
}
```

---

## 4. Emergency Response Escalation UI & Payload Structure

When an `EMERGENCY` risk is detected, the API returns a structured escalation payload:

```json
{
  "safety_status": "EMERGENCY_TRIGGERED",
  "risk_level": "EMERGENCY",
  "trigger_category": "anaphylaxis",
  "message": "EMERGENCY: Immediate Medical Attention Required",
  "directives": [
    "Call emergency medical services immediately.",
    "If available and prescribed, use an Epinephrine auto-injector (EpiPen) for severe allergic reactions.",
    "Do not wait for online advice. Go to the nearest emergency room."
  ],
  "emergency_contacts": [
    {"region": "India", "service": "National Emergency", "number": "112"},
    {"region": "India", "service": "Ambulance", "number": "108"},
    {"region": "India", "service": "National Poison Information Centre (AIIMS)", "number": "1800-116-117"},
    {"region": "United States", "service": "Emergency", "number": "911"},
    {"region": "United States", "service": "Poison Help", "number": "1-800-222-1222"},
    {"region": "International", "service": "Suicide / Crisis Lifeline", "number": "988 / Befrienders Worldwide"}
  ],
  "bypassed_llm": true
}
```

---

## 5. Post-Generation Safety Verification & Audit Logging

1. **Output Safety Filter**:
   * Inspects LLM-generated tokens for forbidden phrases: (e.g., "I diagnose you with", "You should stop taking", "Take a double dose").
   * If detected, the response is redacted and replaced with safe physician-directed guidance.
2. **Audit Logging & Privacy**:
   * Safety incidents trigger anonymized security audit logs (`database/models.py: AuditLog`).
   * No personally identifiable health information (PHI) is written into unencrypted debug logs.
