"""Evaluation utilities and helper functions for Phase 9 Clinical Safety & ML Evaluation."""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a dataset or artifact file."""
    if not file_path.exists():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON Lines dataset safely."""
    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation dataset missing at {file_path}")
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    """Save benchmark results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
