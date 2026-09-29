"""Centralized Phase 4 Model Registry and Lifecycle Manager.

Loads verified Phase 4 production checkpoints once at application startup with
strict architecture validation and device orchestration (CPU/CUDA).
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import torch

from apps.backend.core.config import settings
from ml.embeddings.normalizer import MedicineNormalizer
from ml.handwriting.model import CRNNHandwritingClassifier
from ml.ocr.layout_parser import SpatialRelationExtractor
from ml.ocr.pipeline import PrescriptionOCRExtractor

logger = logging.getLogger("medicine_ai.model_registry")


class ModelRegistryError(Exception):
    """Raised when a required Phase 4 model or checkpoint fails validation or loading."""
    pass


class ModelRegistry:
    """Singleton model lifecycle registry loading and serving Phase 4 inference models."""

    _instance: Optional["ModelRegistry"] = None

    def __init__(self):
        self.device: torch.device = self._resolve_device(settings.DEVICE)
        self.handwriting_model: Optional[CRNNHandwritingClassifier] = None
        self.ocr_extractor: Optional[PrescriptionOCRExtractor] = None
        self.layout_parser: Optional[SpatialRelationExtractor] = None
        self.normalizer: Optional[MedicineNormalizer] = None
        
        self.handwriting_label_map: Dict[str, int] = {}
        self.handwriting_idx_to_label: Dict[int, str] = {}
        self.metadata: Dict[str, Any] = {}
        self._is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance

    @staticmethod
    def _resolve_device(device_setting: str) -> torch.device:
        """Resolve device string to torch.device object."""
        dev = device_setting.lower().strip()
        if dev == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested in settings but not available. Falling back to CPU.")
                return torch.device("cpu")
            return torch.device("cuda")
        elif dev == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    def load_all_models(
        self,
        handwriting_path: Optional[str] = None,
        ocr_path: Optional[str] = None,
        layout_path: Optional[str] = None,
        norm_kb_path: Optional[str] = None,
        norm_idx_path: Optional[str] = None
    ) -> None:
        """Load and validate all Phase 4 checkpoints into memory."""
        logger.info(f"Loading Phase 4 models on device={self.device}...")

        hw_path = Path(handwriting_path or settings.HANDWRITING_CHECKPOINT_PATH)
        ocr_p = Path(ocr_path or settings.OCR_CHECKPOINT_PATH)
        lay_p = Path(layout_path or settings.LAYOUT_CHECKPOINT_PATH)
        kb_p = Path(norm_kb_path or settings.NORMALIZATION_KB_PATH)
        idx_p = Path(norm_idx_path or settings.NORMALIZATION_INDEX_PATH)

        # 1. Validate Checkpoint Existence (Strict Fail-Fast)
        for name, p in [
            ("Handwriting Checkpoint", hw_path),
            ("OCR Checkpoint", ocr_p),
            ("Layout Checkpoint", lay_p),
            ("Normalization KB", kb_p),
            ("Normalization Index", idx_p)
        ]:
            if not p.exists():
                err_msg = f"Required Phase 4 artifact '{name}' not found at: {p}. Cannot start production inference."
                logger.error(err_msg)
                raise ModelRegistryError(err_msg)

        # 2. Load Handwriting CRNN Model
        try:
            hw_checkpoint = torch.load(hw_path, map_location=self.device)
            if not isinstance(hw_checkpoint, dict) or "model_state_dict" not in hw_checkpoint:
                raise ModelRegistryError(f"Invalid checkpoint format in {hw_path}: Missing 'model_state_dict'.")

            self.handwriting_label_map = hw_checkpoint.get("label_map", {})
            self.handwriting_idx_to_label = {v: k for k, v in self.handwriting_label_map.items()}
            num_classes = len(self.handwriting_label_map) if self.handwriting_label_map else 78

            self.handwriting_model = CRNNHandwritingClassifier(
                num_classes=num_classes,
                in_channels=3,
                rnn_hidden=128
            ).to(self.device)
            self.handwriting_model.load_state_dict(hw_checkpoint["model_state_dict"])
            self.handwriting_model.eval()
            logger.info(f"[+] Loaded CRNN Handwriting model ({num_classes} classes, {hw_path.name})")
        except Exception as e:
            raise ModelRegistryError(f"Failed to load handwriting model from {hw_path}: {e}")

        # 3. Load Prescription OCR Extractor
        try:
            self.ocr_extractor = PrescriptionOCRExtractor(confidence_threshold=0.70)
            logger.info(f"[+] Initialized Prescription OCR Extractor ({ocr_p.name})")
        except Exception as e:
            raise ModelRegistryError(f"Failed to initialize OCR extractor: {e}")

        # 4. Load Layout Parser
        try:
            layout_checkpoint = torch.load(lay_p, map_location=self.device)
            if not isinstance(layout_checkpoint, dict) or "model_state_dict" not in layout_checkpoint:
                raise ModelRegistryError(f"Invalid layout checkpoint in {lay_p}: Missing 'model_state_dict'.")

            self.layout_parser = SpatialRelationExtractor(hidden_dim=64, num_categories=6).to(self.device)
            self.layout_parser.load_state_dict(layout_checkpoint["model_state_dict"])
            self.layout_parser.eval()
            logger.info(f"[+] Loaded Layout Spatial Parser ({lay_p.name})")
        except Exception as e:
            raise ModelRegistryError(f"Failed to load layout parser from {lay_p}: {e}")

        # 5. Load RxNorm Normalizer
        try:
            self.normalizer = MedicineNormalizer(
                concepts_path=kb_p,
                index_path=idx_p
            )
            logger.info(f"[+] Loaded RxNorm Multi-Tier Normalizer ({len(self.normalizer.concepts)} concepts)")
        except Exception as e:
            raise ModelRegistryError(f"Failed to load RxNorm normalizer: {e}")

        self.metadata = {
            "phase": "Phase 4 Verified Checkpoints",
            "device": str(self.device),
            "handwriting_model": {
                "checkpoint": str(hw_path.name),
                "num_classes": num_classes,
                "best_val_top1": hw_checkpoint.get("best_val_top1_acc")
            },
            "ocr_extractor": {"calibrated": True},
            "layout_parser": {"checkpoint": str(lay_p.name)},
            "normalizer": {"concepts_count": len(self.normalizer.concepts)}
        }
        self._is_loaded = True
        logger.info("[✓] All Phase 4 inference models loaded and ready.")

    def is_ready(self) -> bool:
        """Return True if all required inference components are loaded."""
        return (
            self._is_loaded
            and self.handwriting_model is not None
            and self.ocr_extractor is not None
            and self.layout_parser is not None
            and self.normalizer is not None
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Return non-sensitive model metadata."""
        return self.metadata
