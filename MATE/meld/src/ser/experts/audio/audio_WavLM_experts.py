# src/ser/experts/audio_universal_npy.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn

from src.ser.experts.registry import register_expert
from src.ser.experts.audio.audio_universal import UniversalAudioExpert

@register_expert("audio_WavLM_experts")
class AudioUniversalNpyExpert(nn.Module):
    """Frame-level audio expert wrapper for saved npy features."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        self.model = UniversalAudioExpert(
            input_dim=int(cfg["input_dim"]),
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            method=str(cfg.get("method", "cnn")),     # cnn / transformer / lstm
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.3)),
            num_classes=int(cfg["num_classes"]),
            kernel_size=int(cfg.get("kernel_size", 3)),
        )

        self.last_pooled: Optional[torch.Tensor] = None
        self.last_attn: Optional[torch.Tensor] = None

    def forward_with_extras(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["x_audio"]
        mask = batch.get("x_audio_mask", None)

        out = self.model(x, mask)  # out={'logits','pooled'}
        logits = out["logits"]
        pooled = out["pooled"]

        attn = getattr(self.model.pooler, "last_attn", None)
        if attn is None:
            attn = torch.zeros((x.shape[0], x.shape[1]), device=x.device)

        self.last_pooled = pooled
        self.last_attn = attn
        return logits, pooled, attn

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        logits, _, _ = self.forward_with_extras(batch)
        return logits
