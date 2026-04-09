from __future__ import annotations

"""
Audio-Visual-Text fusion expert: LMF (Low-rank Multimodal Fusion) over pooled representations.

Pipeline:
  1) per-modality projection -> optional modality-specific Transformer encoder
  2) attentive stats pooling per modality (weighted mean + std)
  3) Low-rank multiplicative fusion (LMF) across (A,V,T) pooled vectors
  4) classifier

Batch keys (expected):
  x_audio, x_audio_mask, x_video, x_video_mask, x_text, x_text_mask, y

forward(batch) -> logits [B,C]
forward_with_extras(batch) -> (logits, pooled, attn_cat)
  - pooled: fused representation [B, fusion_out_dim]
  - attn_cat: concatenated pooling attn [B, Ta_max + Tv_max + Tt_max]
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.ser.experts.registry import register_expert


def _key_padding(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    # torch Transformer/MHA uses True for padding positions
    if mask is None:
        return None
    return (mask == 0)


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


def _pad_attn(attn: torch.Tensor, T_max: int) -> torch.Tensor:
    """Pad attention [B,T] to [B,T_max] with zeros."""
    B, T = attn.shape
    if T == T_max:
        return attn
    out = attn.new_zeros((B, T_max))
    out[:, :T] = attn
    return out


class LowRankFusion(nn.Module):
    """
    Low-rank multimodal fusion.
    For modalities x1..xm:
      z = sum_{r=1..R}  Π_i ( [xi;1] @ Wi[r] )   + bias

    This implementation supports arbitrary number of modalities via a factor list.
    """

    def __init__(self, in_dims: Tuple[int, ...], out_dim: int, rank: int = 8, dropout: float = 0.0):
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.rank = rank
        self.dropout = nn.Dropout(dropout)

        self.factors = nn.ParameterList()
        for d in in_dims:
            # [rank, d+1, out_dim]
            w = nn.Parameter(torch.empty(rank, d + 1, out_dim))
            nn.init.xavier_uniform_(w)
            self.factors.append(w)

        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, xs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # xs: each [B, Di]
        B = xs[0].shape[0]
        device = xs[0].device
        dtype = xs[0].dtype

        fused = None
        for x, w in zip(xs, self.factors):
            ones = torch.ones((B, 1), device=device, dtype=dtype)
            x1 = torch.cat([x, ones], dim=1)  # [B, Di+1]
            # [B, rank, out_dim]
            proj = torch.einsum("bk,rko->bro", x1, w)
            fused = proj if fused is None else (fused * proj)

        # sum over rank -> [B,out_dim]
        y = fused.sum(dim=1) + self.bias
        y = self.dropout(y)
        return y


@register_expert("fusion_avt_lmf")
class FusionAVTLMFExpert(nn.Module):
    """
    Config keys (expert):

    Required:
      - num_classes
      - audio_input_dim
      - video_input_dim
      - text_input_dim

    Recommended:
      - hidden_dim (256)
      - dropout (0.3)
      - use_pos_enc (true)
      - num_layers_audio / num_layers_video / num_layers_text (1)
      - nhead (4)
      - ffn_dim (1024)
      - attn_hidden (128)

      - lmf_rank (8)
      - fusion_out_dim (256)

    Notes:
      - LMF runs on pooled stats vectors: (2H audio) + (2H video) + (2H text)
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

        self.la = int(cfg.get("num_layers_audio", 1))
        self.lv = int(cfg.get("num_layers_video", 1))
        self.lt = int(cfg.get("num_layers_text", 1))
        self.nhead = int(cfg.get("nhead", 4))
        self.ffn_dim = int(cfg.get("ffn_dim", self.h * 4))

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

        def _make_encoder(num_layers: int) -> Optional[nn.TransformerEncoder]:
            if num_layers <= 0:
                return None
            layer = nn.TransformerEncoderLayer(
                d_model=self.h,
                nhead=self.nhead,
                dim_feedforward=self.ffn_dim,
                dropout=self.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=num_layers)

        self.enc_a = _make_encoder(self.la)
        self.enc_v = _make_encoder(self.lv)
        self.enc_t = _make_encoder(self.lt)

        self.pos = SinPosEnc(self.h, max_len=int(cfg.get("pos_max_len", 4000))) if self.use_pos else None

        attn_hidden = int(cfg.get("attn_hidden", 128))
        self.pool_a = AttnStatsPooling(self.h, attn_hidden=attn_hidden, dropout=self.dropout)
        self.pool_v = AttnStatsPooling(self.h, attn_hidden=attn_hidden, dropout=self.dropout)
        self.pool_t = AttnStatsPooling(self.h, attn_hidden=attn_hidden, dropout=self.dropout)

        # LMF on pooled vectors (2H each)
        self.lmf_rank = int(cfg.get("lmf_rank", 8))
        self.fusion_out_dim = int(cfg.get("fusion_out_dim", self.h))

        self.fuse = LowRankFusion(
            in_dims=(self.h * 2, self.h * 2, self.h * 2),
            out_dim=self.fusion_out_dim,
            rank=self.lmf_rank,
            dropout=self.dropout,
        )

        self.cls = nn.Sequential(
            nn.LayerNorm(self.fusion_out_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.fusion_out_dim, self.num_classes),
        )

        self.last_pooled: Optional[torch.Tensor] = None
        self.last_attn: Optional[torch.Tensor] = None

    def _encode_mod(self, x: torch.Tensor, mask: Optional[torch.Tensor], proj: nn.Module, enc: Optional[nn.Module]) -> torch.Tensor:
        h = proj(x)
        if self.pos is not None:
            h = self.pos(h)
        if enc is not None:
            h = enc(h, src_key_padding_mask=_key_padding(mask))
        return h

    def forward_with_extras(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xa = batch["x_audio"]; ma = batch.get("x_audio_mask", None)
        xv = batch["x_video"]; mv = batch.get("x_video_mask", None)
        xt = batch["x_text"];  mt = batch.get("x_text_mask", None)

        ha = self._encode_mod(xa, ma, self.proj_a, self.enc_a)  # [B,Ta,H]
        hv = self._encode_mod(xv, mv, self.proj_v, self.enc_v)  # [B,Tv,H]
        ht = self._encode_mod(xt, mt, self.proj_t, self.enc_t)  # [B,Tt,H]

        pa, attn_a = self.pool_a(ha, ma)  # [B,2H], [B,Ta]
        pv, attn_v = self.pool_v(hv, mv)  # [B,2H], [B,Tv]
        pt, attn_t = self.pool_t(ht, mt)  # [B,2H], [B,Tt]

        fused = self.fuse((pa, pv, pt))   # [B,F]
        logits = self.cls(fused)

        # concatenated attention for analysis
        Ta_max, Tv_max, Tt_max = xa.shape[1], xv.shape[1], xt.shape[1]
        attn_cat = torch.cat([
            _pad_attn(attn_a, Ta_max),
            _pad_attn(attn_v, Tv_max),
            _pad_attn(attn_t, Tt_max),
        ], dim=1)

        self.last_pooled = fused
        self.last_attn = attn_cat
        return logits, fused, attn_cat

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        logits, _, _ = self.forward_with_extras(batch)
        return logits
