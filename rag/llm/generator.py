"""Grounded Medical LLM Response Generator with Evidence Attribution."""
import logging
from typing import Any, Dict, List, Optional

from rag.schemas import (
    Citation,
    DrugInteractionResult,
    IntentType,
    MedicalEntity,
    RetrievalResult,
    SafetyAssessment
)

logger = logging.getLogger("medicine_ai.rag.llm")


class MedicalLLM:
    """Evidence-grounded medical text generation engine."""

    def __init__(self, backend: str = "deterministic"):
        self.backend = backend

    def build_system_prompt(self) -> str:
        """Construct strict clinical guardrail system prompt."""
        return (
            "You are an authoritative, evidence-grounded AI Clinical Pharmacist Assistant.\n"
            "STRICT CLINICAL RULES:\n"
            "1. Answer ONLY using the facts presented in the provided RETRIEVED EVIDENCE.\n"
            "2. Do NOT invent, assume, or extrapolate any drug dosages, indications, side effects, or interactions.\n"
            "3. If evidence is missing or insufficient, state clearly: 'Information on this topic is not available in the verified clinical database.'\n"
            "4. For every substantive medical statement, cite the corresponding source using [Source: chunk_id].\n"
            "5. Never prescribe medication or give individual diagnostic opinions.\n"
            "6. Always distinguish verified facts from uncertain or unverified prescription handwriting.\n"
        )

    def generate_response(
        self,
        query: str,
        intent: IntentType,
        safety: SafetyAssessment,
        entities: List[MedicalEntity],
        evidence: List[RetrievalResult],
        interactions: List[DrugInteractionResult],
        prescription_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Synthesize clinical response grounded strictly in retrieved evidence and structured interactions."""
        # 1. Check if safety guardrails blocked generation
        if not safety.is_safe_to_generate and safety.emergency_guidance:
            return safety.emergency_guidance

        # 2. Check for out-of-scope query
        if safety.safety_level == "OUT_OF_SCOPE" or intent == "OUT_OF_SCOPE":
            return (
                "I am specialized as an AI Clinical Medicine Assistant. I can assist with verified prescription details, "
                "authoritative drug monographs (indications, dosage guidelines, adverse effects, contraindications), "
                "and structured drug-drug interaction safety checks. Please ask a medicine- or prescription-related question."
            )

        response_parts: List[str] = []

        # 3. Handle Prescription Context (Preserving Phase 5 Verification Policies)
        if prescription_context:
            candidates = prescription_context.get("candidates", [])
            verified_meds = [c for c in candidates if c.get("verification_status") == "verified"]
            unverified_meds = [c for c in candidates if c.get("verification_status") in ["unverified", "review_required"]]

            if verified_meds:
                med_names = [v.get("normalized_name") or v.get("raw_text") for v in verified_meds]
                response_parts.append(
                    f"**Prescription Context (Verified)**: Found confirmed medication(s): {', '.join(med_names)}."
                )

            if unverified_meds:
                unv_names = [u.get("raw_text") for u in unverified_meds]
                response_parts.append(
                    f"⚠️ **Prescription Warning**: The following item(s) from your prescription could not be reliably verified "
                    f"({', '.join(unv_names)}). Please have your prescribing physician or a licensed pharmacist inspect the original prescription before taking any medication."
                )

        # 4. Handle Structured Drug-Drug Interactions
        if intent == "DRUG_INTERACTION" or interactions:
            if interactions:
                response_parts.append("### 🔍 Verified Drug Interaction Assessment:")
                for inter in interactions:
                    if inter.interaction_found:
                        response_parts.append(
                            f"- **{inter.drug_a.upper()} + {inter.drug_b.upper()}** [{inter.severity}]:\n"
                            f"  - **Clinical Effect**: {inter.clinical_effect}\n"
                            f"  - **Mechanism**: {inter.mechanism}\n"
                            f"  - **Clinical Recommendation**: {inter.recommendation}\n"
                            f"  *(Evidence Source: {inter.evidence_source})*"
                        )
                    else:
                        response_parts.append(
                            f"- **{inter.drug_a.upper()} + {inter.drug_b.upper()}**: {inter.clinical_effect} "
                            f"{inter.recommendation}"
                        )
            elif intent == "DRUG_INTERACTION" and not interactions and len(entities) >= 2:
                drug_entities = [e.canonical_name or e.text for e in entities if e.label in ["DRUG", "BRAND", "ACTIVE_INGREDIENT"]]
                response_parts.append(
                    f"### 🔍 Drug Interaction Safety Check:\n"
                    f"No verified drug interaction was found in the authoritative clinical database for the combination: "
                    f"{' + '.join(drug_entities)}. Always consult your doctor or pharmacist to confirm safety before co-administering medications."
                )

        # 5. Grounded Evidence Monograph Synthesis
        if evidence:
            response_parts.append("### 📋 Clinical Monograph Information:")
            for doc in evidence[:3]:
                sec = doc.section_name or "Clinical Overview"
                chunk_id_tag = f"[Source: {doc.chunk_id or doc.document_id}]"
                response_parts.append(
                    f"**{doc.title} — {sec}** {chunk_id_tag}:\n"
                    f"{doc.text.strip()}\n"
                )
        elif not interactions and not prescription_context:
            # Missing evidence fallback
            detected_drugs = [e.canonical_name or e.text for e in entities if e.label in ["DRUG", "BRAND", "ACTIVE_INGREDIENT"]]
            if detected_drugs:
                response_parts.append(
                    f"Information regarding '{', '.join(detected_drugs)}' for this specific clinical inquiry is not available in the verified clinical monograph database. "
                    f"Please consult a healthcare professional or licensed pharmacist."
                )
            else:
                response_parts.append(
                    "No verified clinical evidence was found matching your inquiry in the authoritative drug monograph repository. "
                    "Please verify the medicine name or consult a qualified healthcare provider."
                )

        # 6. Append Clinical Disclaimer
        response_parts.append(
            "\n---\n*Disclaimer: This information is derived from official drug labeling (DailyMed / RxNorm) for educational purposes only. "
            "Never alter medication doses or initiate treatment without consulting a licensed healthcare professional.*"
        )

        return "\n\n".join(response_parts)
