from __future__ import annotations

"""Audio-Visual-Text fusion expert: Cross-Attention + Shared Transformer + Attentive Stats Pooling.

Tri-modal extension of fusion_av_xattn.

Design:
  1) per-modality projection -> modality-specific transformer encoder
  2) tri-directional cross-attention:
        A attends to (V,T),
        V attends to (A,T),
        T attends to (A,V),
     each with residual + layer norm
  3) token-level shared transformer over concatenated (A,V,T) tokens
  4) attentive statistics pooling + linear classifier

Batch keys (expected):
  x_audio, x_audio_mask, x_video, x_video_mask, x_text, x_text_mask, y
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
        s = self.net(x).squeeze(-1)  # [B,T]
        if mask is not None:
            # guard: avoid NaN when all tokens are masked
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

def _ensure_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.new_ones((x.shape[0], x.shape[1]), dtype=torch.long)
    return mask

def _cat_kv(h1: torch.Tensor, m1: Optional[torch.Tensor], h2: torch.Tensor, m2: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concat K/V sequences and masks along time."""
    m1 = _ensure_mask(h1, m1)
    m2 = _ensure_mask(h2, m2)
    return torch.cat([h1, h2], dim=1), torch.cat([m1, m2], dim=1)

@register_expert("fusion_avt_xattn")
class FusionAVTCrossAttnExpert(nn.Module):
    """Config keys (expert):

    Required:
      - num_classes
      - audio_input_dim
      - video_input_dim
      - text_input_dim

    Recommended:
      - hidden_dim (256)
      - nhead (4)
      - ffn_dim (1024)
      - num_layers_audio (2)
      - num_layers_video (2)
      - num_layers_text (2)
      - num_layers_shared (2)
      - dropout (0.4)
      - attn_hidden (128)
      - use_pos_enc (true)
      - modality_emb (true)
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        self.num_classes = int(cfg["num_classes"])
        self.da = int(cfg["audio_input_dim"])
        self.dv = int(cfg["video_input_dim"])
        self.dt = int(cfg["text_input_dim"])

        self.h = int(cfg.get("hidden_dim", 256))
        self.nhead = int(cfg.get("nhead", 4))
        self.ffn_dim = int(cfg.get("ffn_dim", self.h * 4))
        self.dropout = float(cfg.get("dropout", 0.4))

        self.la = int(cfg.get("num_layers_audio", 2))
        self.lv = int(cfg.get("num_layers_video", 2))
        self.lt = int(cfg.get("num_layers_text", 2))
        self.ls = int(cfg.get("num_layers_shared", 2))

        self.use_pos = bool(cfg.get("use_pos_enc", True))
        self.use_mod_emb = bool(cfg.get("modality_emb", True))

        # --- per-modality projection ---
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

        # --- modality-specific encoders ---
        def _make_encoder(num_layers: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=self.h,
                nhead=self.nhead,
                dim_feedforward=self.ffn_dim,
                dropout=self.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=max(1, num_layers))

        self.enc_a = _make_encoder(self.la)
        self.enc_v = _make_encoder(self.lv)
        self.enc_t = _make_encoder(self.lt)
        self.enc_shared = _make_encoder(self.ls)

        self.pos = SinPosEnc(self.h, max_len=int(cfg.get("pos_max_len", 4000))) if self.use_pos else None

        # --- cross-attention ---
        self.xattn_a2vt = nn.MultiheadAttention(self.h, self.nhead, dropout=self.dropout, batch_first=True)
        self.xattn_v2at = nn.MultiheadAttention(self.h, self.nhead, dropout=self.dropout, batch_first=True)
        self.xattn_t2av = nn.MultiheadAttention(self.h, self.nhead, dropout=self.dropout, batch_first=True)

        self.xattn_drop = nn.Dropout(self.dropout)
        self.ln_a = nn.LayerNorm(self.h)
        self.ln_v = nn.LayerNorm(self.h)
        self.ln_t = nn.LayerNorm(self.h)

        # --- modality embeddings for shared transformer ---
        if self.use_mod_emb:
            self.mod_a = nn.Parameter(torch.zeros(1, 1, self.h))
            self.mod_v = nn.Parameter(torch.zeros(1, 1, self.h))
            self.mod_t = nn.Parameter(torch.zeros(1, 1, self.h))
            nn.init.normal_(self.mod_a, std=0.02)
            nn.init.normal_(self.mod_v, std=0.02)
            nn.init.normal_(self.mod_t, std=0.02)
        else:
            self.mod_a = None
            self.mod_v = None
            self.mod_t = None

        # --- pooling + classifier ---
        attn_hidden = int(cfg.get("attn_hidden", 128))
        self.pool = AttnStatsPooling(self.h, attn_hidden=attn_hidden, dropout=self.dropout)
        self.cls = nn.Sequential(
            nn.LayerNorm(self.h * 2),
            nn.Dropout(self.dropout),
            nn.Linear(self.h * 2, self.num_classes),
        )

        self.last_pooled: Optional[torch.Tensor] = None
        self.last_attn: Optional[torch.Tensor] = None

    def _encode_mod(self, x: torch.Tensor, mask: Optional[torch.Tensor], proj: nn.Module, enc: nn.Module) -> torch.Tensor:
        h = proj(x)
        if self.pos is not None:
            h = self.pos(h)
        return enc(h, src_key_padding_mask=_key_padding(mask))

    def forward_with_extras(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xa = batch["x_audio"]; ma = batch.get("x_audio_mask", None)
        xv = batch["x_video"]; mv = batch.get("x_video_mask", None)
        xt = batch["x_text"];  mt = batch.get("x_text_mask", None)

        # 1) modality-specific encoders
        ha = self._encode_mod(xa, ma, self.proj_a, self.enc_a)  # [B,Ta,H]
        hv = self._encode_mod(xv, mv, self.proj_v, self.enc_v)  # [B,Tv,H]
        ht = self._encode_mod(xt, mt, self.proj_t, self.enc_t)  # [B,Tt,H]

        # 2) tri-directional cross-attention + residual
        hv_t, mv_t = _cat_kv(hv, mv, ht, mt)
        ha2, _ = self.xattn_a2vt(query=ha, key=hv_t, value=hv_t, key_padding_mask=_key_padding(mv_t))
        ha = self.ln_a(ha + self.xattn_drop(ha2))
        if ma is not None:
            ha = ha.masked_fill(ma.unsqueeze(-1) == 0, 0.0)

        ha_t, ma_t = _cat_kv(ha, ma, ht, mt)
        hv2, _ = self.xattn_v2at(query=hv, key=ha_t, value=ha_t, key_padding_mask=_key_padding(ma_t))
        hv = self.ln_v(hv + self.xattn_drop(hv2))
        if mv is not None:
            hv = hv.masked_fill(mv.unsqueeze(-1) == 0, 0.0)

        ha_v, ma_v = _cat_kv(ha, ma, hv, mv)
        ht2, _ = self.xattn_t2av(query=ht, key=ha_v, value=ha_v, key_padding_mask=_key_padding(ma_v))
        ht = self.ln_t(ht + self.xattn_drop(ht2))
        if mt is not None:
            ht = ht.masked_fill(mt.unsqueeze(-1) == 0, 0.0)

        # 3) shared encoder over concatenated tokens
        if self.use_mod_emb:
            ha = ha + self.mod_a
            hv = hv + self.mod_v
            ht = ht + self.mod_t

        ma_ = _ensure_mask(xa, ma)
        mv_ = _ensure_mask(xv, mv)
        mt_ = _ensure_mask(xt, mt)

        x_cat = torch.cat([ha, hv, ht], dim=1)
        m_cat = torch.cat([ma_, mv_, mt_], dim=1)

        h = self.enc_shared(x_cat, src_key_padding_mask=_key_padding(m_cat))

        pooled, attn = self.pool(h, mask=m_cat)
        logits = self.cls(pooled)

        self.last_pooled = pooled
        self.last_attn = attn
        return logits, pooled, attn

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        logits, _, _ = self.forward_with_extras(batch)
        return logits
