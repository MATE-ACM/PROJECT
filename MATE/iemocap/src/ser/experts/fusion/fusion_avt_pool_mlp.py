from __future__ import annotations

"""Audio-Visual-Text fusion expert: Attentive Stats Pooling + MLP.

Tri-modal extension of the AV pooling baseline. It is intentionally simple and
serves as a classic late-fusion control with a clear inductive bias.

Batch keys (expected):
  x_audio, x_audio_mask,
  x_video, x_video_mask,
  x_text,  x_text_mask,
  y

forward(batch) -> logits [B,C]
forward_with_extras(batch) -> (logits, pooled, extras)
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.ser.experts.registry import register_expert


class SinPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(dtype=x.dtype, device=x.device)


class AttnStatsPooling(nn.Module):
    """Attentive statistics pooling: weighted mean + std -> [B, 2H]."""

    def __init__(self, hidden_dim: int, attn_hidden: int = 128, dropout: float = 0.0, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, 1),
        )
        self.last_attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,T,H]; mask: [B,T] 1=valid
        s = self.net(x).squeeze(-1)  # [B,T]
        if mask is not None:
            # guard: if a sample is fully masked, force the first token valid to avoid NaN
            all0 = (mask.sum(dim=1) == 0)
            if torch.any(all0):
                mask = mask.clone()
                mask[all0, 0] = 1
            s = s.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(s, dim=1)
        self.last_attn = attn
        w = attn.unsqueeze(-1)
        mu = torch.sum(w * x, dim=1)
        ex2 = torch.sum(w * (x * x), dim=1)
        var = (ex2 - mu * mu).clamp_min(0.0)
        std = torch.sqrt(var + self.eps)
        pooled = torch.cat([mu, std], dim=-1)
        return pooled, attn


def _key_padding(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return (mask == 0)


@register_expert("fusion_avt_pool_mlp")
class FusionAVTPoolMLPExpert(nn.Module):
    """
    Late fusion tri-modal baseline:
      proj -> (optional tiny per-mod encoder) -> attentive stats pooling per modality
      -> concat -> MLP -> logits
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        self.num_classes = int(cfg["num_classes"])
        self.da = int(cfg["audio_input_dim"])
        self.dv = int(cfg["video_input_dim"])
        self.dt = int(cfg["text_input_dim"])

        self.h = int(cfg.get("hidden_dim", 256))
        self.dropout = float(cfg.get("dropout", 0.3))
        self.use_pos = bool(cfg.get("use_pos_enc", True))

        # projections
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
        self.proj_t = nn.Sequential(
            nn.Linear(self.dt, self.h),
            nn.LayerNorm(self.h),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        # optional tiny modality encoders (cheap)
        num_layers_mod = int(cfg.get("num_layers_mod", 1))
        if num_layers_mod > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=self.h,
                nhead=int(cfg.get("nhead", 4)),
                dim_feedforward=int(cfg.get("ffn_dim", self.h * 4)),
                dropout=self.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.enc_a = nn.TransformerEncoder(layer, num_layers=num_layers_mod)
            self.enc_v = nn.TransformerEncoder(layer, num_layers=num_layers_mod)
            self.enc_t = nn.TransformerEncoder(layer, num_layers=num_layers_mod)
        else:
            self.enc_a = None
            self.enc_v = None
            self.enc_t = None

        self.pos = SinPosEnc(self.h, max_len=int(cfg.get("pos_max_len", 4000))) if self.use_pos else None

        attn_hidden = int(cfg.get("attn_hidden", 128))
        self.pool_a = AttnStatsPooling(self.h, attn_hidden=attn_hidden, dropout=self.dropout)
        self.pool_v = AttnStatsPooling(self.h, attn_hidden=attn_hidden, dropout=self.dropout)
        self.pool_t = AttnStatsPooling(self.h, attn_hidden=attn_hidden, dropout=self.dropout)

        # pooled_a/v/t are [B,2H] => concat [B,6H]
        fused_dim = self.h * 6
        mlp_dim = int(cfg.get("mlp_dim", self.h * 3))
        self.cls = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(self.dropout),
            nn.Linear(fused_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(mlp_dim, self.num_classes),
        )

        self.last_pooled: Optional[torch.Tensor] = None

    def _encode(self, x: torch.Tensor, mask: Optional[torch.Tensor], proj: nn.Module, enc: Optional[nn.Module]) -> torch.Tensor:
        h = proj(x)
        if self.pos is not None:
            h = self.pos(h)
        if enc is not None:
            h = enc(h, src_key_padding_mask=_key_padding(mask))
        return h

    def forward_with_extras(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        xa = batch["x_audio"]; ma = batch.get("x_audio_mask", None)
        xv = batch["x_video"]; mv = batch.get("x_video_mask", None)
        xt = batch["x_text"];  mt = batch.get("x_text_mask", None)

        ha = self._encode(xa, ma, self.proj_a, self.enc_a)
        hv = self._encode(xv, mv, self.proj_v, self.enc_v)
        ht = self._encode(xt, mt, self.proj_t, self.enc_t)

        pa, attn_a = self.pool_a(ha, ma)  # [B,2H]
        pv, attn_v = self.pool_v(hv, mv)  # [B,2H]
        pt, attn_t = self.pool_t(ht, mt)  # [B,2H]

        pooled = torch.cat([pa, pv, pt], dim=-1)  # [B,6H]
        logits = self.cls(pooled)
        self.last_pooled = pooled

        extras = {"attn_audio": attn_a, "attn_video": attn_v, "attn_text": attn_t}
        return logits, pooled, extras

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        logits, _, _ = self.forward_with_extras(batch)
        return logits
