"""Document layout understanding, spatial bounding-box parsing, and key-value linking."""
import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialRelationExtractor(nn.Module):
    """Neural spatial relation classifier predicting Key-Value linking between bounding boxes."""

    def __init__(self, bbox_dim: int = 4, hidden_dim: int = 64, num_categories: int = 6):
        super().__init__()
        # Input features: (bbox1_normalized[4], bbox2_normalized[4], spatial_delta[4]) -> 12 dims
        self.fc = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_categories) # 0: none, 1: doctor, 2: patient, 3: age, 4: date, 5: rx
        )

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        return self.fc(spatial_features)


def extract_spatial_features(bbox1: List[int], bbox2: List[int], page_w: int = 1000, page_h: int = 1000) -> torch.Tensor:
    """Compute 12-dimensional spatial and geometric features between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = [c / max(page_w, page_h) for c in bbox1]
    x2_min, y2_min, x2_max, y2_max = [c / max(page_w, page_h) for c in bbox2]

    dx = x2_min - x1_max # horizontal gap
    dy = y2_min - y1_max # vertical gap
    dw = (x2_max - x2_min) - (x1_max - x1_min)
    dh = (y2_max - y2_min) - (y1_max - y1_min)

    feat = [
        x1_min, y1_min, x1_max, y1_max,
        x2_min, y2_min, x2_max, y2_max,
        dx, dy, dw, dh
    ]
    return torch.tensor(feat, dtype=torch.float32)
