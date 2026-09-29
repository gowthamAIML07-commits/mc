"""Medical Named Entity Recognition (NER) and Clinical Entity Extractor."""
import re
from typing import Dict, List, Optional
from ml.embeddings.normalizer import MedicineNormalizer
from rag.schemas import MedicalEntity


class MedicalEntityExtractor:
    """Extracts clinical entities (drugs, dosages, symptoms, diseases) and normalizes drug tokens."""

    def __init__(self, normalizer: Optional[MedicineNormalizer] = None):
        self.normalizer = normalizer or MedicineNormalizer()

    def extract_entities(self, text: str) -> List[MedicalEntity]:
        """Detect and extract medical entities with character spans and clinical normalization."""
        entities: List[MedicalEntity] = []

        # 1. Detect Dosage, Strengths & Quantities
        dosage_matches = list(re.finditer(r"\b(\d+(?:\.\d+)?\s*(?:mg|mcg|g|gm|ml|iu|%|tablets?|capsules?|drops?))\b", text, re.IGNORECASE))
        for m in dosage_matches:
            entities.append(MedicalEntity(
                text=m.group(1),
                label="DOSAGE",
                start_char=m.start(),
                end_char=m.end(),
                confidence=0.98
            ))

        # 2. Detect Frequencies & Administration Schedules
        freq_matches = list(re.finditer(r"\b(1-0-1|1-0-0|0-0-1|1-1-1|OD|BD|BID|TDS|TID|QID|QDS|SOS|HS|STAT|PRN|twice daily|once daily|three times daily|before meals|after meals|at bedtime)\b", text, re.IGNORECASE))
        for m in freq_matches:
            entities.append(MedicalEntity(
                text=m.group(1),
                label="FREQUENCY",
                start_char=m.start(),
                end_char=m.end(),
                confidence=0.95
            ))

        # 3. Detect Duration
        dur_matches = list(re.finditer(r"\b(\d+\s*(?:days?|weeks?|months?|d|wks?))\b", text, re.IGNORECASE))
        for m in dur_matches:
            # avoid overlapping with dosage
            if not any(e.start_char <= m.start() and m.end() <= e.end_char for e in entities):
                entities.append(MedicalEntity(
                    text=m.group(1),
                    label="DURATION",
                    start_char=m.start(),
                    end_char=m.end(),
                    confidence=0.95
                ))

        # 4. Detect Clinical Symptoms & Disease Conditions
        conditions_vocab = {
            "fever": "SYMPTOM", "headache": "SYMPTOM", "cough": "SYMPTOM", "chest pain": "SYMPTOM",
            "shortness of breath": "SYMPTOM", "diarrhea": "SYMPTOM", "nausea": "SYMPTOM", "vomiting": "SYMPTOM",
            "rash": "SYMPTOM", "itching": "SYMPTOM", "dizziness": "SYMPTOM", "pain": "SYMPTOM",
            "hypertension": "DISEASE", "diabetes": "DISEASE", "type 2 diabetes": "DISEASE",
            "asthma": "DISEASE", "copd": "DISEASE", "pneumonia": "DISEASE", "infection": "DISEASE",
            "sinusitis": "DISEASE", "gerd": "DISEASE", "acid reflux": "DISEASE", "pharyngitis": "DISEASE",
            "anaphylaxis": "SYMPTOM", "allergic reaction": "ALLERGY", "penicillin allergy": "ALLERGY"
        }
        for term, label in conditions_vocab.items():
            for m in re.finditer(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
                entities.append(MedicalEntity(
                    text=m.group(0),
                    label=label,
                    start_char=m.start(),
                    end_char=m.end(),
                    confidence=0.95
                ))

        # 5. Detect & Normalize Drug / Brand / Active Ingredient Mentions
        # Additional clinical vocabulary for monographs & interaction rules
        clinical_drugs_vocab = {
            "warfarin": ("11289", "Warfarin"),
            "aspirin": ("1191", "Aspirin"),
            "clarithromycin": ("2551", "Clarithromycin"),
            "spironolactone": ("9997", "Spironolactone"),
            "lisinopril": ("29046", "Lisinopril"),
            "tramadol": ("10689", "Tramadol"),
            "fluoxetine": ("4493", "Fluoxetine"),
            "methotrexate": ("6851", "Methotrexate"),
            "alcohol": (None, "Alcohol"),
            "coumadin": ("11289", "Warfarin (Coumadin)"),
            "advil": ("310965", "Ibuprofen (Advil)"),
            "tylenol": ("161", "Paracetamol (Tylenol)")
        }

        for drug_term, (known_cui, known_canonical) in clinical_drugs_vocab.items():
            for m in re.finditer(rf"\b{re.escape(drug_term)}\b", text, re.IGNORECASE):
                if not any(e.start_char <= m.start() and m.end() <= e.end_char for e in entities if e.label in ["DRUG", "BRAND", "ACTIVE_INGREDIENT"]):
                    entities.append(MedicalEntity(
                        text=m.group(0),
                        label="DRUG",
                        start_char=m.start(),
                        end_char=m.end(),
                        normalized_rxcui=known_cui,
                        canonical_name=known_canonical,
                        confidence=0.96
                    ))

        # Check against normalizer concepts and aliases
        NON_DRUG_WORDS = {
            "the", "and", "for", "with", "take", "tab", "cap", "syr", "day", "days",
            "how", "do", "write", "python", "function", "to", "sort", "list", "what",
            "where", "when", "why", "who", "which", "is", "are", "was", "were", "can",
            "could", "should", "would", "tell", "explain", "about", "give", "help",
            "this", "that", "these", "those", "have", "has", "had", "from", "into",
            "over", "after", "before", "between", "under", "above", "such", "than",
            "then", "very", "much", "many", "more", "most", "some", "any", "not",
            "only", "own", "same", "will", "just", "code", "file", "test", "time"
        }
        text_words = re.findall(r"\b[A-Za-z0-9\-]+\b", text)
        for word in text_words:
            if len(word) < 3 or word.lower() in NON_DRUG_WORDS:
                continue

            norm = self.normalizer.normalize(word)
            tier = norm.get("match_tier", "none")
            conf = norm.get("confidence", 0)
            if norm.get("rxnorm_id") and (tier in ["exact", "exact_clean", "exact_alias", "clinical_abbreviation", "abbreviation", "phonetic"] or conf >= 0.85):
                # Locate span in text
                for m in re.finditer(rf"\b{re.escape(word)}\b", text, re.IGNORECASE):
                    # Check not already annotated
                    if not any(e.start_char <= m.start() and m.end() <= e.end_char for e in entities if e.label in ["DRUG", "BRAND", "ACTIVE_INGREDIENT"]):
                        entities.append(MedicalEntity(
                            text=m.group(0),
                            label="DRUG",
                            start_char=m.start(),
                            end_char=m.end(),
                            normalized_rxcui=norm.get("rxnorm_id"),
                            canonical_name=norm.get("normalized_name"),
                            confidence=norm.get("confidence", 0.90)
                        ))

        entities.sort(key=lambda x: x.start_char)
        return entities
