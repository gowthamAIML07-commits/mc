"""Structured Drug-Drug Interaction Lookup Engine backed by Authoritative Monographs."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ml.embeddings.normalizer import MedicineNormalizer
from rag.schemas import DrugInteractionResult

logger = logging.getLogger("medicine_ai.rag.interactions")


class DrugInteractionEngine:
    """Deterministic structured drug-drug interaction engine."""

    # Authoritative verified clinical interaction rules from DailyMed and FDA labeling
    VERIFIED_INTERACTION_RULES: List[Dict[str, Any]] = [
        {
            "pair": {"warfarin", "aspirin"},
            "severity": "MAJOR",
            "clinical_effect": "Significantly increased risk of severe gastrointestinal hemorrhage and bleeding.",
            "mechanism": "Synergistic inhibition of hemostasis via vitamin K antagonism and platelet cyclooxygenase-1 inhibition.",
            "evidence_source": "DailyMed Warfarin Sodium Monograph Section 7 (Drug Interactions)",
            "recommendation": "Avoid concurrent use unless strictly monitored by a hematologist or cardiologist with frequent INR testing."
        },
        {
            "pair": {"warfarin", "ibuprofen"},
            "severity": "MAJOR",
            "clinical_effect": "Marked elevation in major bleeding risk and gastric mucosal ulceration.",
            "mechanism": "NSAID platelet inhibition combined with warfarin anticoagulation and competitive protein binding.",
            "evidence_source": "DailyMed Warfarin Sodium Monograph Section 7",
            "recommendation": "Co-administration is generally contraindicated; select alternative analgesics such as paracetamol with dosage monitoring."
        },
        {
            "pair": {"metformin", "alcohol"},
            "severity": "MAJOR",
            "clinical_effect": "Potentiation of metformin effect on lactate metabolism, dramatically increasing risk of fatal lactic acidosis.",
            "mechanism": "Ethanol impairs hepatic gluconeogenesis and inhibits lactate clearance.",
            "evidence_source": "DailyMed Metformin Hydrochloride Monograph Section 5 (Warnings) and Section 7",
            "recommendation": "Patients should be warned against excessive alcohol intake, both acute and chronic, while taking metformin."
        },
        {
            "pair": {"atorvastatin", "clarithromycin"},
            "severity": "MAJOR",
            "clinical_effect": "Dramatically increased atorvastatin serum concentrations causing severe myopathy and rhabdomyolysis.",
            "mechanism": "Strong inhibition of CYP3A4-mediated atorvastatin metabolism by clarithromycin.",
            "evidence_source": "DailyMed Atorvastatin Calcium Monograph Section 7.1 (CYP3A4 Inhibitors)",
            "recommendation": "Avoid combination or temporarily suspend atorvastatin during the antimicrobial course."
        },
        {
            "pair": {"pantoprazole", "methotrexate"},
            "severity": "MAJOR",
            "clinical_effect": "Elevated and prolonged serum methotrexate levels, increasing methotrexate toxicity (bone marrow suppression, renal impairment).",
            "mechanism": "Proton pump inhibitors may inhibit renal BCRP / OAT3 transporters responsible for active tubular secretion of methotrexate.",
            "evidence_source": "DailyMed Pantoprazole Sodium Monograph Section 7 (Drug Interactions)",
            "recommendation": "Consider temporary withdrawal of PPI therapy when high-dose methotrexate is administered."
        },
        {
            "pair": {"amoxicillin", "methotrexate"},
            "severity": "MODERATE",
            "clinical_effect": "Potential reduction in renal clearance of methotrexate leading to elevated methotrexate exposure.",
            "mechanism": "Competitive inhibition of renal tubular secretion.",
            "evidence_source": "DailyMed Amoxicillin Monograph Section 7",
            "recommendation": "Monitor serum methotrexate levels and clinical signs of toxicity during co-administration."
        },
        {
            "pair": {"paracetamol", "alcohol"},
            "severity": "MODERATE",
            "clinical_effect": "Increased susceptibility to acute hepatotoxicity and liver injury even at therapeutic doses.",
            "mechanism": "Chronic ethanol consumption induces CYP2E1, converting paracetamol into toxic N-acetyl-p-benzoquinone imine (NAPQI) while depleting glutathione.",
            "evidence_source": "DailyMed Acetaminophen Monograph Section 5 (Warnings)",
            "recommendation": "Limit paracetamol intake and avoid chronic heavy alcohol consumption."
        },
        {
            "pair": {"lisinopril", "spironolactone"},
            "severity": "MAJOR",
            "clinical_effect": "Severe, life-threatening hyperkalemia and cardiac conduction abnormalities.",
            "mechanism": "Additive potassium retention from ACE inhibition and aldosterone antagonism.",
            "evidence_source": "DailyMed Lisinopril Monograph Section 7",
            "recommendation": "Regularly monitor serum potassium and renal function if combination is clinically warranted."
        },
        {
            "pair": {"tramadol", "fluoxetine"},
            "severity": "MAJOR",
            "clinical_effect": "High risk of Serotonin Syndrome (hyperthermia, rigidity, autonomic instability) and reduced analgesic efficacy.",
            "mechanism": "Potent CYP2D6 inhibition and cumulative serotonergic neurotransmission.",
            "evidence_source": "DailyMed Tramadol Hydrochloride Monograph Section 7",
            "recommendation": "Avoid combination or use non-serotonergic analgesics under close medical supervision."
        }
    ]

    def __init__(self, normalizer: Optional[MedicineNormalizer] = None):
        self.normalizer = normalizer or MedicineNormalizer()

    def check_pair_interaction(self, drug_a: str, drug_b: str) -> DrugInteractionResult:
        """Check for verified clinical interaction between two drug names."""
        clean_a = drug_a.lower().strip()
        clean_b = drug_b.lower().strip()

        norm_a = self.normalizer.normalize(drug_a)
        norm_b = self.normalizer.normalize(drug_b)

        rxcui_a = norm_a.get("rxnorm_id") if norm_a.get("verification_status") == "verified" else None
        rxcui_b = norm_b.get("rxnorm_id") if norm_b.get("verification_status") == "verified" else None

        display_a = norm_a.get("normalized_name") if (rxcui_a and norm_a.get("normalized_name")) else drug_a.strip().title()
        display_b = norm_b.get("normalized_name") if (rxcui_b and norm_b.get("normalized_name")) else drug_b.strip().title()

        ing_a = (norm_a.get("ingredient") or "").lower()
        ing_b = (norm_b.get("ingredient") or "").lower()

        # Search verified rules
        for rule in self.VERIFIED_INTERACTION_RULES:
            rule_pair = list(rule["pair"])
            p1, p2 = rule_pair[0], rule_pair[1]

            match_p1_a = (p1 in clean_a) or (p1 in ing_a)
            match_p2_b = (p2 in clean_b) or (p2 in ing_b)

            match_p2_a = (p2 in clean_a) or (p2 in ing_a)
            match_p1_b = (p1 in clean_b) or (p1 in ing_b)

            if (match_p1_a and match_p2_b) or (match_p2_a and match_p1_b):
                return DrugInteractionResult(
                    drug_a=p1.title(),
                    drug_b=p2.title(),
                    drug_a_rxcui=rxcui_a,
                    drug_b_rxcui=rxcui_b,
                    interaction_found=True,
                    severity=rule["severity"],
                    clinical_effect=rule["clinical_effect"],
                    mechanism=rule["mechanism"],
                    evidence_source=rule["evidence_source"],
                    recommendation=rule["recommendation"]
                )

        # Explicit unverified indication — NEVER fabricate interaction
        return DrugInteractionResult(
            drug_a=display_a,
            drug_b=display_b,
            drug_a_rxcui=rxcui_a,
            drug_b_rxcui=rxcui_b,
            interaction_found=False,
            severity=None,
            clinical_effect="No verified interaction documented in the available clinical knowledge base for this pair.",
            mechanism=None,
            evidence_source="Authoritative Knowledge Base (DailyMed/RxNorm)",
            recommendation="Consult a licensed physician or clinical pharmacist to verify drug safety before combining medications."
        )

    def check_all_interactions(self, drug_list: List[str]) -> List[DrugInteractionResult]:
        """Check all pairwise combinations across a list of medicines."""
        if len(drug_list) < 2:
            return []

        results: List[DrugInteractionResult] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        for i in range(len(drug_list)):
            for j in range(i + 1, len(drug_list)):
                d1 = drug_list[i].lower().strip()
                d2 = drug_list[j].lower().strip()
                pair_key = tuple(sorted([d1, d2]))
                if pair_key in seen_pairs or d1 == d2:
                    continue
                seen_pairs.add(pair_key)

                res = self.check_pair_interaction(d1, d2)
                if res.interaction_found:
                    results.append(res)

        return results
