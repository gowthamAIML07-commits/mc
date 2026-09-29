"""Authoritative Master Clinical Drug Dataset (500+ Concepts) for Phase 10A Expansion.

Derived strictly from official FDA DailyMed, US NLM RxNorm, and FDA Approved Drug Product labeling.
Consolidates 14 major clinical therapeutic classes with deterministic provenance, RxCUIs, and section metadata.
"""
import logging
from typing import Any, Dict, List

from rag.ingestion.domains_abx_pain_cv import get_abx_pain_cv_records
from rag.ingestion.domains_metabolic_gi_resp import get_metabolic_gi_resp_records
from rag.ingestion.domains_psych_immuno_cns import get_psych_immuno_cns_records
from rag.ingestion.domains_specialty_onco import get_specialty_onco_records
from rag.ingestion.domains_expanded_catalog import get_expanded_catalog_records

logger = logging.getLogger("medicine_ai.rag.dataset")

def load_extended_catalog() -> List[Dict[str, Any]]:
    """Assemble, deduplicate, and return the complete authoritative clinical drug catalog.
    
    Returns:
        List of distinct medicine dictionary records anchored by RxNorm RxCUIs.
    """
    raw_records: List[Dict[str, Any]] = []
    
    # 1. Domains 1-3: Antibiotics, Analgesics/Musculoskeletal, Cardiovascular
    raw_records.extend(get_abx_pain_cv_records())
    
    # 2. Domains 4-8: Antidiabetic, Lipids, Anticoagulant/Antiplatelet, GI, Respiratory
    raw_records.extend(get_metabolic_gi_resp_records())
    
    # 3. Domains 9-11: Corticosteroids/Immuno, Psychiatric/CNS, Antifungals/Antivirals
    raw_records.extend(get_psych_immuno_cns_records())
    
    # 4. Domains 12-14: Urological/Bone, Oncology/Hematology, Dermatology/Ophthalmic
    raw_records.extend(get_specialty_onco_records())
    
    # 5. Expanded Catalog: Extended Antivirals, Psych, Oncology, Special Care
    raw_records.extend(get_expanded_catalog_records())
    
    # Deduplicate deterministically by normalized lowercase name
    unique_catalog: List[Dict[str, Any]] = []
    seen_names = set()
    
    for record in raw_records:
        name_key = record.get("name", "").strip().lower()
        if not name_key:
            continue
        if name_key in seen_names:
            # Skip duplicate occurrences, preserving earliest comprehensive definition
            continue
        seen_names.add(name_key)
        unique_catalog.append(record)
        
    logger.info(
        "Master Clinical Dataset loaded: %d total inputs reduced to %d unique concepts.",
        len(raw_records), len(unique_catalog)
    )
    return unique_catalog

# Global reference for fast access
MASTER_DRUG_CATALOG: List[Dict[str, Any]] = load_extended_catalog()

if __name__ == "__main__":
    catalog = load_extended_catalog()
    print(f"Total Unique Verified Clinical Concepts: {len(catalog)}")
    rxcuis = set(d.get("rxcui") for d in catalog if d.get("rxcui"))
    print(f"Total Distinct RxCUIs: {len(rxcuis)}")
