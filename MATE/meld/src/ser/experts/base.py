from __future__ import annotations

"""Base interfaces for expert models."""

from typing import Dict, Any
import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
