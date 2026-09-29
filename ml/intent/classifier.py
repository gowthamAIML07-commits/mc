"""Clinical Intent Classification Engine for Medical Inquiries."""
import re
from typing import Dict, List, Tuple
from rag.schemas import IntentType


class MedicalIntentClassifier:
    """Classifies user medical messages into structured intent taxonomy."""

    INTENT_PATTERNS: Dict[IntentType, List[str]] = {
        "EMERGENCY": [
            r"\b(chest pain|heart attack|can't breathe|cannot breathe|suffocating|severe allergic reaction|anaphylaxis|passed out|unconscious|overdose|drank poison|swallowed poison|suicide|bleeding heavily|stroke symptoms)\b"
        ],
        "DRUG_INTERACTION": [
            r"\b(interactions?|interact(?:ing)?|together\s+with|take\s+with|combine\s+with|safe\s+with|mix\s+with|taken\s+along\s+with)\b",
            r"\b(take\s+[A-Za-z0-9\s\-]+\s+(?:together\s+)?with)\b",
            r"\b(safe\s+to\s+take\s+[A-Za-z0-9\s\-]+\s+with)\b",
            r"\b(can\s+i\s+take\s+[A-Za-z0-9\s\-]+\s+(?:together\s+)?with)\b",
            r"\b(can\s+i\s+take\s+[A-Za-z0-9\s\-]+\s+and\s+[A-Za-z0-9\s\-]+)\b"
        ],
        "PREGNANCY": [
            r"\b(pregnant|pregnancy|during pregnancy|trimester|breastfeeding|nursing mother|lactation|baby in womb)\b"
        ],
        "CHILD_MEDICATION": [
            r"\b(child(?:ren)?|infant|pediatric|baby|toddler|for kids|safe for children|in kids)\b"
        ],
        "ELDERLY_MEDICATION": [
            r"\b(elderly|senior citizen|old age|geriatric|dose for elderly)\b"
        ],
        "MISSED_DOSE": [
            r"\b(missed dose|forgot to take|forgot my pill|skipped dose|missed my tablet)\b"
        ],
        "CONTRAINDICATION": [
            r"\b(contraindications?|who\s+should\s+not\s+take|avoid\s+if|when\s+not\s+to\s+take|unsafe\s+for|warnings?\s+for|dangerous\s+if)\b"
        ],
        "SIDE_EFFECT": [
            r"\b(side[\s\-]effects?|adverse[\s\-]reactions?|bad[\s\-]reactions?|causes?\s+headache|causes?\s+nausea|harmful\s+effects?|toxicity|vomiting\s+after\s+taking)\b"
        ],
        "DOSAGE_INFORMATION": [
            r"\b(how\s+much\s+to\s+take|how\s+many\s+tablets?|dosages?|doses?|maximum\s+dose|how\s+often|frequency|take\s+before\s+or\s+after\s+food|take\s+with\s+meals?)\b"
        ],
        "MEDICINE_IDENTIFICATION": [
            r"\b(identify|what is this pill|what medicine is this|name of this drug|unclear handwriting|recognize)\b"
        ],
        "PRESCRIPTION": [
            r"\b(prescription|my prescription|doctor prescribed|rx details|read my prescription)\b"
        ],
        "MEDICINE_USE": [
            r"\b(what is it used for|what does it treat|why is it given|indicated for|prescribed for|usage of)\b"
        ],
        "SYMPTOM": [
            r"\b(i have fever|my throat hurts|coughing|headache|stomach pain|acid reflux|symptoms of)\b"
        ],
        "MEDICINE_INFORMATION": [
            r"\b(tell me about|information on|details about|what is|monograph|drug info)\b"
        ]
    }

    def classify_intent(self, text: str) -> IntentType:
        """Deterministically classify inquiry into clinical intent."""
        text_lower = text.lower().strip()

        # Check in priority order (Emergency first)
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, text_lower, re.IGNORECASE):
                    return intent

        # Fallback heuristic
        if any(w in text_lower for w in ["health", "diet", "lifestyle", "exercise"]):
            return "GENERAL_HEALTH"

        if len(text_lower.split()) < 2:
            return "OUT_OF_SCOPE"

        return "MEDICINE_INFORMATION"
