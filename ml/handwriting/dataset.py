"""PyTorch Dataset loader for Doctor's Handwritten Prescription BD dataset."""
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class HandwrittenMedicineDataset(Dataset):
    """Dataset loader for isolated handwritten medicine word crops."""

    def __init__(
        self,
        base_dir: Path,
        split: str = "Training",
        img_size: Tuple[int, int] = (64, 192),
        label_map: Optional[Dict[str, int]] = None
    ):
        self.base_dir = Path(base_dir)
        self.split = split
        self.img_size = img_size # (height, width)
        self.samples: List[Tuple[Path, str, str]] = [] # (img_path, med_name, generic_name)
        self.label_map: Dict[str, int] = label_map or {}
        
        self._load_split_data()

    def _load_split_data(self):
        """Find CSV labels and word images for the specified split."""
        split_candidates = [
            self.base_dir / f"Doctor's Handwritten Prescription BD dataset/{self.split}",
            self.base_dir / f"Doctor_s Handwritten Prescription BD dataset/{self.split}",
            self.base_dir / self.split,
        ]
        
        split_dir = None
        for cand in split_candidates:
            if cand.exists():
                split_dir = cand
                break
                
        if not split_dir:
            # Search recursively for directories matching split name
            found = [d for d in self.base_dir.rglob(f"*{self.split}*") if d.is_dir()]
            if found:
                split_dir = found[0]

        if not split_dir or not split_dir.exists():
            raise FileNotFoundError(f"Split directory for '{self.split}' not found under {self.base_dir}")

        csv_files = list(split_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Labels CSV not found in {split_dir}")

        csv_path = csv_files[0]
        words_dirs = [d for d in split_dir.iterdir() if d.is_dir() and "word" in d.name.lower()]
        words_dir = words_dirs[0] if words_dirs else split_dir

        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get("IMAGE") or row.get("image")
                med_name = row.get("MEDICINE_NAME") or row.get("medicine_name") or "Unknown"
                gen_name = row.get("GENERIC_NAME") or row.get("generic_name") or ""
                if img_name:
                    img_path = words_dir / img_name
                    if img_path.exists():
                        self.samples.append((img_path, med_name.strip(), gen_name.strip()))

        # Build label map if not provided
        if not self.label_map:
            unique_meds = sorted(list(set(s[1] for s in self.samples)))
            self.label_map = {name: idx for idx, name in enumerate(unique_meds)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, med_name, _ = self.samples[idx]
        
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = img.resize((self.img_size[1], self.img_size[0])) # (width, height)
                arr = np.array(img, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1) # (C, H, W)
        except Exception:
            tensor = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)

        label_idx = self.label_map.get(med_name, 0)
        return tensor, label_idx, med_name
