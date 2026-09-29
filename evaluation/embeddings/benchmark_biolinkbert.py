"""Benchmark wrapper for Model C: BioLinkBERT (Link-aware biomedical language model)."""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

proj_root = Path(__file__).resolve().parent.parent.parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from evaluation.embeddings.base import BaseEmbeddingBenchmarkModel

logger = logging.getLogger("medicine_ai.eval.biolinkbert")


class BioLinkBERTEmbeddingModel(BaseEmbeddingBenchmarkModel):
    """Model C: BioLinkBERT (michiyasunaga/BioLinkBERT-base)."""

    DEFAULT_CHECKPOINT = "michiyasunaga/BioLinkBERT-base"

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_CHECKPOINT,
        dimension: int = 768,
        max_seq_length: int = 128,
        device: Optional[str] = None
    ):
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(
            model_name="biolinkbert",
            dimension=dimension,
            max_seq_length=max_seq_length,
            device=dev
        )
        self.checkpoint_id = model_name_or_path
        logger.info(f"Loading BioLinkBERT tokenizer and weights from {self.checkpoint_id} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_id, local_files_only=True)
            self.model = AutoModel.from_pretrained(self.checkpoint_id, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_id)
            self.model = AutoModel.from_pretrained(self.checkpoint_id)
        self.model.to(self.device)
        self.model.eval()

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a single batch of texts using BioLinkBERT attention-weighted mean pooling."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Attention-weighted mean pooling across token representations
            input_mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask, dim=1)
            sum_mask = torch.clamp(input_mask.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            norm_repr = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

        return norm_repr.cpu().numpy().astype(np.float32)

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode clinical texts in batches."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embs = self._encode_batch(batch)
            all_embs.append(embs)

        return np.vstack(all_embs)

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries using BioLinkBERT."""
        return self.encode_texts(queries, batch_size=32)

    def get_model_metadata(self) -> Dict[str, Any]:
        """Return deterministic model configuration."""
        import transformers
        return {
            "model_name": "BioLinkBERT-base",
            "model_identifier": self.checkpoint_id,
            "model_revision": "main",
            "embedding_dimension": self.dimension,
            "tokenizer": "BertTokenizer (WordPiece, vocab=28895)",
            "max_sequence_length": self.max_seq_length,
            "pooling": "[CLS] Token Embedding",
            "normalization": "L2 Normalization (unit sphere)",
            "device": self.device,
            "license": "MIT License / Open Access",
            "library_versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "numpy": np.__version__
            }
        }
