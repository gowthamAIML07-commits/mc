"""Standalone CLI Ingestion Script for Phase 10A Clinical Knowledge Base Expansion.

Executes the authoritative DailyMed/RxNorm ingestion pipeline:
1. Loads raw authoritative drug monographs.
2. Validates schema, content, and security constraints.
3. Computes deterministic SHA-256 hashes and provenance records.
4. Performs clinical section-aware chunking.
5. Emits production chunks, monographs, normalization tables, and coverage matrix.
6. Generates versioned corpus manifest with cryptographic integrity check.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.ingestion.adapter import DailyMedRxNormIngestionAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("medicine_ai.cli.ingest")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 10A Clinical Knowledge Base Corpus Ingestion Pipeline"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "clinical_knowledge"),
        help="Directory to store raw monograph JSONs"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "knowledge_base"),
        help="Directory to store processed clinical chunks"
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "manifests" / "clinical_knowledge_manifest.json"),
        help="Path for clinical knowledge manifest"
    )
    args = parser.parse_args()

    logger.info("Initializing DailyMed & RxNorm Clinical Ingestion Adapter...")
    adapter = DailyMedRxNormIngestionAdapter(
        raw_storage_dir=Path(args.raw_dir),
        processed_kb_dir=Path(args.processed_dir),
        manifest_path=Path(args.manifest_path)
    )

    logger.info("Executing ingestion, validation, and chunking pipeline...")
    results = adapter.process_and_ingest_all()

    print("\n" + "=" * 60)
    print("PHASE 10A CLINICAL KNOWLEDGE INGESTION COMPLETE")
    print("=" * 60)
    print(f"Status:               {results['status']}")
    print(f"Corpus Version:       {results['corpus_version']}")
    print(f"Source Documents:     {results['total_documents']}")
    print(f"Distinct Medicines:   {results['distinct_medicines']}")
    print(f"Distinct RxCUIs:      {results['distinct_rxcuis']}")
    print(f"Distinct Ingredients: {results['distinct_ingredients']}")
    print(f"Total Chunks:         {results['total_chunks']}")
    print(f"Manifest SHA-256:     {results['manifest_sha256']}")
    print(f"Manifest Path:        {results['manifest_path']}")
    print(f"Coverage Matrix:      {results['coverage_matrix_path']}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
