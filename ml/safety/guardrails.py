"""Medical Safety Classifier and Clinical Guardrails Engine."""
import re
from typing import List, Optional
from rag.schemas import MedicalEntity, SafetyAssessment, SafetyLevel


class MedicalSafetyGuardrails:
    """Clinical guardrails engine that evaluates inquiries for emergency risks, prescribing attempts, and safety."""

    EMERGENCY_TRIGGERS = [
        r"\b(chest pain|heart attack|myocardial infarction|can'?t breathe|cannot breathe|suffocating|difficulty breathing|shortness of breath)\b",
        r"\b(anaphylaxis|severe allergic reaction|throat swelling|swollen tongue and lips|stridor)\b",
        r"\b(overdose|took too many pills|swallowed whole bottle|poisoning|drank poison|ingested poison)\b",
        r"\b(stroke|face drooping|slurred speech|arm weakness|sudden numbness)\b",
        r"\b(unconscious|passed out|loss of consciousness|unresponsive|seizure|convulsions)\b",
        r"\b(severe bleeding|coughing up blood|vomiting blood|heavy hemorrhage)\b",
        r"\b(suicide|kill myself|end my life|self harm|want to die)\b"
    ]

    PRESCRIBING_TRIGGERS = [
        r"\b(prescribe me|give me a prescription|write me an rx|write a prescription for|can you prescribe)\b",
        r"\b(how much should i take to cure|diagnose me|do i have cancer|tell me what illness i have)\b"
    ]

    OUT_OF_SCOPE_TRIGGERS = [
        r"\b(python|javascript|code|programming|recipe|weather|cricket|football|stock market|cryptocurrency|investing)\b"
    ]

    EMERGENCY_RESPONSE_TEXT = (
        "🚨 **CRITICAL MEDICAL ALERT / IMMEDIATE ATTENTION REQUIRED**:\n\n"
        "Your message indicates a potentially life-threatening medical emergency or severe acute condition.\n\n"
        "**IMMEDIATE ACTION REQUIRED:**\n"
        "- Call your local emergency services immediately (such as **911** in the US/Canada, **112** in the EU/India, or **999** in the UK).\n"
        "- Proceed to the nearest hospital Emergency Department / Urgent Care Center.\n"
        "- If poison or overdose is suspected, contact your local Poison Control Center immediately.\n\n"
        "Do not rely on this AI assistant for emergency medical advice or acute clinical intervention."
    )

    def assess_safety(
        self,
        text: str,
        intent: Optional[str] = None,
        entities: Optional[List[MedicalEntity]] = None
    ) -> SafetyAssessment:
        """Evaluate medical safety risks, emergency status, and clinical boundaries."""
        text_lower = text.lower().strip()
        safety_flags: List[str] = []

        # 1. Check Life-Threatening Emergency Conditions
        for pattern in self.EMERGENCY_TRIGGERS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                safety_flags.append(f"EMERGENCY_TRIGGER_DETECTED: {pattern}")
                return SafetyAssessment(
                    safety_level="EMERGENCY",
                    is_emergency=True,
                    is_safe_to_generate=False,
                    safety_flags=safety_flags,
                    emergency_guidance=self.EMERGENCY_RESPONSE_TEXT
                )

        if intent == "EMERGENCY":
            safety_flags.append("INTENT_EMERGENCY_CLASSIFIED")
            return SafetyAssessment(
                safety_level="EMERGENCY",
                is_emergency=True,
                is_safe_to_generate=False,
                safety_flags=safety_flags,
                emergency_guidance=self.EMERGENCY_RESPONSE_TEXT
            )

        # 2. Check Prescribing or Self-Diagnosis Requests (High Risk)
        for pattern in self.PRESCRIBING_TRIGGERS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                safety_flags.append("PRESCRIBING_OR_DIAGNOSIS_ATTEMPT")
                return SafetyAssessment(
                    safety_level="HIGH",
                    is_emergency=False,
                    is_safe_to_generate=True,
                    safety_flags=safety_flags,
                    emergency_guidance=None
                )

        # 3. Check Out-of-Scope Queries
        for pattern in self.OUT_OF_SCOPE_TRIGGERS:
            if re.search(pattern, text_lower, re.IGNORECASE) and not any(e.label in ["DRUG", "SYMPTOM", "DISEASE"] for e in (entities or [])):
                safety_flags.append("NON_MEDICAL_OUT_OF_SCOPE")
                return SafetyAssessment(
                    safety_level="OUT_OF_SCOPE",
                    is_emergency=False,
                    is_safe_to_generate=True,
                    safety_flags=safety_flags,
                    emergency_guidance=None
                )

        # 4. Check Moderate-Risk Contexts (Pregnancy, Pediatric, Interactions, Dosages)
        if intent in ["PREGNANCY", "CHILD_MEDICATION", "DRUG_INTERACTION", "DOSAGE_INFORMATION", "CONTRAINDICATION"]:
            safety_flags.append(f"SPECIAL_POPULATION_OR_RISK_TOPIC_{intent}")
            return SafetyAssessment(
                safety_level="MODERATE",
                is_emergency=False,
                is_safe_to_generate=True,
                safety_flags=safety_flags,
                emergency_guidance=None
            )

        # 5. Low Risk Informational Query
        return SafetyAssessment(
            safety_level="LOW",
            is_emergency=False,
            is_safe_to_generate=True,
            safety_flags=safety_flags,
            emergency_guidance=None
        )
