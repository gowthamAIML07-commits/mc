"""Hardware, environment, and dataset provenance utilities."""
import hashlib
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch


def get_hardware_info() -> Dict[str, Any]:
    """Capture comprehensive hardware, OS, and framework environment specifications."""
    info = {
        "os_name": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    return info


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest for a single file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_directory_sha256(dir_path: Path, max_files: int = 100) -> str:
    """Compute deterministic SHA-256 hash over directory contents."""
    if not dir_path.exists():
        return "DIR_NOT_FOUND"

    sha256_hash = hashlib.sha256()
    all_files = sorted([p for p in dir_path.rglob("*") if p.is_file()])[:max_files]
    
    if not all_files:
        return "EMPTY_DIRECTORY"

    for file_path in all_files:
        rel_path = file_path.relative_to(dir_path).as_posix()
        sha256_hash.update(rel_path.encode("utf-8"))
        try:
            with open(file_path, "rb") as f:
                sha256_hash.update(f.read(8192))
        except Exception:
            continue

    return sha256_hash.hexdigest()
