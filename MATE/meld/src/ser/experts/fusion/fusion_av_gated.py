from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.ser.experts.registry import register_expert

class AttnPooling(nn.Module):
    """Attentive mean pooling -> [B,H]"""

    def __init__(self, hidden_dim: int, attn_hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, 1),
        )
        self.last_attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.net(x).squeeze(-1)  # [B,T]
        if mask is not None:
            all0 = (mask.sum(dim=1) == 0)
            if torch.any(all0):
                mask = mask.clone()
                mask[all0, 0] = 1
            s = s.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(s, dim=1)  # [B,T]
        self.last_attn = attn
        h = torch.sum(attn.unsqueeze(-1) * x, dim=1)  # [B,H]
        return h, attn

@register_expert("fusion_av_gated")
class FusionAVGatedExpert(nn.Module):
    """
    proj -> per-mod attentive mean pooling -> gate -> fused -> classifier
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        self.num_classes = int(cfg["num_classes"])
        self.da = int(cfg["audio_input_dim"])
        self.dv = int(cfg["video_input_dim"])

        self.h = int(cfg.get("hidden_dim", 256))
        self.dropout = float(cfg.get("dropout", 0.3))

        self.proj_a = nn.Sequential(
            nn.Linear(self.da, self.h),
            nn.LayerNorm(self.h),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.proj_v = nn.Sequential(
            nn.Linear(self.dv, self.h),
            nn.LayerNorm(self.h),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        attn_hidden = int(cfg.get("attn_hidden", 128))
        self.pool_a = AttnPooling(self.h, attn_hidden=attn_hidden, dropout=self.dropout)
        self.pool_v = AttnPooling(self.h, attn_hidden=attn_hidden, dropout=self.dropout)

        # gate: g in (0,1)^H  (vector gate, not scalar)
        self.gate = nn.Sequential(
            nn.Linear(self.h * 2, self.h),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.h, self.h),
            nn.Sigmoid(),
        )

        self.cls = nn.Sequential(
            nn.LayerNorm(self.h),
            nn.Dropout(self.dropout),
            nn.Linear(self.h, self.num_classes),
        )

        self.last_gate: Optional[torch.Tensor] = None
        self.last_attn_a: Optional[torch.Tensor] = None
        self.last_attn_v: Optional[torch.Tensor] = None

    def forward_with_extras(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        xa = batch["x_audio"]
        ma = batch.get("x_audio_mask", None)
        xv = batch["x_video"]
        mv = batch.get("x_video_mask", None)

        ha = self.proj_a(xa)
        hv = self.proj_v(xv)

        pa, attn_a = self.pool_a(ha, ma)  # [B,H]
        pv, attn_v = self.pool_v(hv, mv)  # [B,H]

        g = self.gate(torch.cat([pa, pv], dim=-1))  # [B,H]
        fused = g * pa + (1.0 - g) * pv  # [B,H]

        logits = self.cls(fused)

        self.last_gate = g
        self.last_attn_a = attn_a
        self.last_attn_v = attn_v
        extras = {"gate": g, "attn_audio": attn_a, "attn_video": attn_v}
        return logits, fused, extras

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        logits, _, _ = self.forward_with_extras(batch)
        return logits
