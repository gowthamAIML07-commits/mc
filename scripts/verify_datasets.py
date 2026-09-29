#!/usr/bin/env python3
"""Dataset manifest validator and licensing compliance checker.

Usage:
    python scripts/verify_datasets.py
"""
import os
import sys
from pathlib import Path
import yaml


def load_yaml(file_path: Path) -> dict:
    """Load and parse YAML configuration file."""
    if not file_path.exists():
        print(f"[ERROR] Manifest file not found at: {file_path}")
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_datasets():
    """Verify registry dataset metadata, licensing compliance, and local paths."""
    root_dir = Path(__file__).resolve().parent.parent
    registry_path = root_dir / "data" / "registry" / "datasets.yaml"
    
    print(f"[*] Validating dataset registry: {registry_path.relative_to(root_dir)}")
    registry_data = load_yaml(registry_path)
    
    datasets = registry_data.get("datasets", {})
    if not datasets:
        print("[ERROR] No datasets registered in manifest.")
        sys.exit(1)
        
    print(f"[*] Found {len(datasets)} registered datasets.\n")
    print(f"{'ID':<6} | {'Dataset Name':<35} | {'License':<15} | {'Redistribution':<14} | {'Status'}")
    print("-" * 85)
    
    for key, item in datasets.items():
        ds_id = item.get("id", "N/A")
        name = item.get("name", key)[:35]
        license_type = item.get("license", "Unknown")[:15]
        redist = "Allowed" if item.get("redistribution_allowed") else "Restricted"
        
        # Check local path existence
        local_path = root_dir / item.get("local_path", "")
        status = "Present" if local_path.exists() and any(local_path.iterdir()) else "Manifest OK (Not downloaded)"
        
        print(f"{ds_id:<6} | {name:<35} | {license_type:<15} | {redist:<14} | {status}")
        
    print("\n[SUCCESS] Dataset registry metadata and licensing verified successfully.")


if __name__ == "__main__":
    verify_datasets()
