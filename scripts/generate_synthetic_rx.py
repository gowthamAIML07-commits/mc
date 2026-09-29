"""Executable CLI script to generate compliant synthetic prescription datasets."""
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.ocr.augmentation import SyntheticPrescriptionGenerator



def generate_synthetic_dataset(
    output_dir: Path,
    train_count: int = 15,
    val_count: int = 5,
    test_count: int = 5,
    seed: int = 42
):
    """Generate isolated synthetic prescription batches across train, val, and test splits."""
    generator = SyntheticPrescriptionGenerator(output_dir=output_dir)

    train_anns = generator.generate_split_batch("train", train_count, seed=seed)
    val_anns = generator.generate_split_batch("val", val_count, seed=seed)
    test_anns = generator.generate_split_batch("test", test_count, seed=seed)

    summary = {
        "dataset_name": "Synthetic Medical Prescription Specimen Dataset",
        "intended_task": "Prescription Augmentation & Layout Stress-Testing (Pipeline C)",
        "license": "Internal Synthetic Data Generation (Permissive / MIT)",
        "synthetic_watermark": SyntheticPrescriptionGenerator.WATERMARK_TEXT,
        "seed": seed,
        "total_samples": train_count + val_count + test_count,
        "splits": {
            "train": {"count": train_count, "annotations_file": str(output_dir / "train" / "annotations.json")},
            "val": {"count": val_count, "annotations_file": str(output_dir / "val" / "annotations.json")},
            "test": {"count": test_count, "annotations_file": str(output_dir / "test" / "annotations.json")}
        },
        "sample_fields": ["prescription_id", "split", "is_synthetic", "disclaimer", "doctor", "patient", "date", "medicines", "instructions"]
    }

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Successfully generated synthetic dataset manifest: {manifest_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic medical prescriptions")
    parser.add_argument("--train_count", type=int, default=15, help="Number of train samples")
    parser.add_argument("--val_count", type=int, default=5, help="Number of val samples")
    parser.add_argument("--test_count", type=int, default=5, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--output_dir", type=str, default="data/processed/synthetic_rx_specimen", help="Output directory")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    target_dir = root_dir / args.output_dir
    generate_synthetic_dataset(
        output_dir=target_dir,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
