"""Multi-Tier Medicine Normalization Engine (RxNorm + RxTerms)."""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import Levenshtein


def double_metaphone_key(word: str) -> str:
    """Clinical phonetic representation for medicine names."""
    w = word.upper().strip()
    w = re.sub(r"[^A-Z]", "", w)
    if not w:
        return ""
    w = re.sub(r"^(KN|GN|PN|AE|WR)", lambda m: m.group(0)[1], w)
    w = re.sub(r"^X", "S", w)
    w = re.sub(r"^PS", "S", w)
    w = re.sub(r"^PH", "F", w)
    w = re.sub(r"PH", "F", w)
    w = re.sub(r"TH", "T", w)
    w = re.sub(r"GH", "G", w)
    w = re.sub(r"CK", "K", w)
    w = re.sub(r"C([EIY])", r"S\1", w)
    w = re.sub(r"C", "K", w)
    w = re.sub(r"Z", "S", w)
    w = re.sub(r"X", "KS", w)
    w = re.sub(r"DG", "J", w)
    w = re.sub(r"QU|Q", "K", w)
    w = re.sub(r"V", "F", w)
    
    first = w[0]
    rest = re.sub(r"[AEIOUY]", "", w[1:])
    dedup = re.sub(r"(.)\1+", r"\1", rest)
    return (first + dedup)[:8]


# Backward compatibility alias
simple_metaphone = double_metaphone_key


def sanitize_term(term: str) -> str:
    """Sanitize term for indexing and matching."""
    t = term.lower().strip()
    t = re.sub(r"^(tab\.|cap\.|syr\.|inj\.|tab|cap|syr|inj)\s*", "", t)
    t = re.sub(r"\b(\d+(\.\d+)?\s*(mg|mcg|g|gm|ml|iu|%))\b", "", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


class MedicineNormalizer:
    """4-Tier Normalization and Verification Engine mapping raw trade names to RxCUIs."""

    def __init__(
        self,
        index_path: Optional[Path] = None,
        concepts_path: Optional[Path] = None,
        rxnorm_path: Optional[Path] = None
    ):
        root_dir = Path(__file__).resolve().parent.parent.parent
        
        target_concepts = rxnorm_path or concepts_path
        if target_concepts is None:
            norm_concepts = root_dir / "data" / "processed" / "normalization" / "normalized_medicines.json"
            if norm_concepts.exists():
                target_concepts = norm_concepts
            else:
                target_concepts = root_dir / "data" / "processed" / "rxnorm" / "rxnorm_concepts.json"

        if index_path is None:
            index_path = root_dir / "data" / "processed" / "normalization" / "lexical_index.json"

        self.concepts_path = Path(target_concepts)
        self.index_path = Path(index_path)


        self.concepts: List[dict] = []
        self.concept_by_rxcui: Dict[str, dict] = {}
        self.exact_map: Dict[str, dict] = {}
        self.phonetic_map: Dict[str, List[dict]] = {}
        self.abbreviations: Dict[str, Dict[str, str]] = {}
        
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Load concepts and indexes."""
        if self.concepts_path.exists():
            with open(self.concepts_path, "r", encoding="utf-8") as f:
                self.concepts = json.load(f)
            self.concept_by_rxcui = {c["rxcui"]: c for c in self.concepts}

        if self.index_path.exists():
            with open(self.index_path, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
                self.abbreviations = idx_data.get("abbreviations", {})

        def _register_exact(key: str, item: dict, is_primary_ingredient: bool = False):
            if not key:
                return
            if key not in self.exact_map:
                self.exact_map[key] = item
            else:
                existing = self.exact_map[key]
                existing_ing = sanitize_term(existing.get("ingredient", ""))
                current_ing = sanitize_term(item.get("ingredient", ""))
                # If current item's ingredient matches the key exactly, prefer single-agent over combinations
                if is_primary_ingredient and current_ing == key and existing_ing != key:
                    self.exact_map[key] = item
                elif current_ing == key and "/" in existing_ing:
                    self.exact_map[key] = item

        # Build in-memory lookup maps
        for item in self.concepts:
            rxcui = item["rxcui"]
            name_lower = item["name"].lower()
            ing_lower = item["ingredient"].lower()
            name_sanitized = sanitize_term(item["name"])
            ing_sanitized = sanitize_term(item["ingredient"])
            
            _register_exact(name_lower, item, is_primary_ingredient=False)
            _register_exact(ing_lower, item, is_primary_ingredient=True)
            _register_exact(name_sanitized, item, is_primary_ingredient=False)
            _register_exact(ing_sanitized, item, is_primary_ingredient=True)

            # Phonetic indexing
            ph_key = double_metaphone_key(ing_sanitized)
            self.phonetic_map.setdefault(ph_key, []).append(item)

            for alias in item.get("aliases", []):
                alias_lower = alias.lower()
                alias_sanitized = sanitize_term(alias)
                _register_exact(alias_lower, item, is_primary_ingredient=False)
                _register_exact(alias_sanitized, item, is_primary_ingredient=False)
                
                alias_ph = double_metaphone_key(alias_sanitized)
                self.phonetic_map.setdefault(alias_ph, []).append(item)

    def normalize(self, raw_query: str, min_confidence: float = 0.65) -> Dict[str, Any]:
        """Normalize noisy OCR string to official RxNorm concept."""
        if not raw_query or not raw_query.strip():
            return {
                "raw_name": raw_query,
                "normalized_name": "Unknown",
                "rxnorm_id": None,
                "confidence": 0.0,
                "verification_status": "unverified",
                "match_tier": "none"
            }

        cleaned = re.sub(r"^(tab\.|cap\.|syr\.|inj\.|tab|cap|syr|inj)\s*", "", raw_query.strip(), flags=re.IGNORECASE)
        cleaned_lower = cleaned.lower()
        cleaned_sanitized = sanitize_term(cleaned)
        query_nums = re.findall(r"\d+", raw_query)

        # Tier 0: Clinical Shorthand / Abbreviation check
        shorthand_dict = self.abbreviations.get("drug_shorthands", {})
        if cleaned_sanitized in shorthand_dict:
            resolved_target = shorthand_dict[cleaned_sanitized].lower()
            if resolved_target in self.exact_map:
                target = self.exact_map[resolved_target]
                return {
                    "raw_name": raw_query,
                    "normalized_name": target["name"],
                    "rxnorm_id": target["rxcui"],
                    "ingredient": target["ingredient"],
                    "strength": target["strength"],
                    "confidence": 0.98,
                    "verification_status": "verified",
                    "match_tier": "abbreviation_resolution"
                }

        # Tier 1: Exact / Alias Match
        if cleaned_lower in self.exact_map:
            target = self.exact_map[cleaned_lower]
            return {
                "raw_name": raw_query,
                "normalized_name": target["name"],
                "rxnorm_id": target["rxcui"],
                "ingredient": target["ingredient"],
                "strength": target["strength"],
                "confidence": 0.98,
                "verification_status": "verified",
                "match_tier": "exact_hash"
            }
        
        if cleaned_sanitized in self.exact_map:
            target = self.exact_map[cleaned_sanitized]
            return {
                "raw_name": raw_query,
                "normalized_name": target["name"],
                "rxnorm_id": target["rxcui"],
                "ingredient": target["ingredient"],
                "strength": target["strength"],
                "confidence": 0.95,
                "verification_status": "verified",
                "match_tier": "exact_sanitized"
            }

        # Tier 2: Phonetic Key Match
        ph_key = double_metaphone_key(cleaned_sanitized)
        candidates = self.phonetic_map.get(ph_key, [])
        if candidates:
            best_cand, best_sim = None, 0.0
            for cand in candidates:
                sim = Levenshtein.ratio(cleaned_sanitized, sanitize_term(cand["ingredient"]))
                sim = max(sim, Levenshtein.ratio(cleaned_sanitized, sanitize_term(cand["name"])))
                for a in cand.get("aliases", []):
                    sim = max(sim, Levenshtein.ratio(cleaned_sanitized, sanitize_term(a)))
                
                # Strength matching bonus
                if query_nums and any(num in cand["strength"] for num in query_nums):
                    sim += 0.20

                if sim > best_sim:
                    best_sim = sim
                    best_cand = cand
            
            if best_cand and best_sim >= 0.60:
                return {
                    "raw_name": raw_query,
                    "normalized_name": best_cand["name"],
                    "rxnorm_id": best_cand["rxcui"],
                    "ingredient": best_cand["ingredient"],
                    "strength": best_cand["strength"],
                    "confidence": min(round(0.80 + 0.18 * best_sim, 2), 0.99),
                    "verification_status": "verified" if best_sim >= 0.75 else "review_required",
                    "match_tier": "phonetic_metaphone"
                }

        # Tier 3: Global Fuzzy Levenshtein Search
        best_cand, best_sim = None, 0.0
        for cand in self.concepts:
            sim = Levenshtein.ratio(cleaned_sanitized, sanitize_term(cand["name"]))
            sim = max(sim, Levenshtein.ratio(cleaned_sanitized, sanitize_term(cand["ingredient"])))
            for a in cand.get("aliases", []):
                sim = max(sim, Levenshtein.ratio(cleaned_sanitized, sanitize_term(a)))
            
            # Strength matching bonus
            if query_nums and any(num in cand["strength"] for num in query_nums):
                sim += 0.20

            if sim > best_sim:
                best_sim = sim
                best_cand = cand

        if best_cand and best_sim >= min_confidence:
            return {
                "raw_name": raw_query,
                "normalized_name": best_cand["name"],
                "rxnorm_id": best_cand["rxcui"],
                "ingredient": best_cand["ingredient"],
                "strength": best_cand["strength"],
                "confidence": min(round(best_sim, 2), 0.99),
                "verification_status": "verified" if best_sim >= 0.85 else "review_required",
                "match_tier": "fuzzy_levenshtein"
            }

        return {
            "raw_name": raw_query,
            "normalized_name": "Unrecognized Entity",
            "rxnorm_id": None,
            "ingredient": "Unknown",
            "strength": "Unknown",
            "confidence": 0.0,
            "verification_status": "unverified",
            "match_tier": "none"
        }
