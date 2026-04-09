from __future__ import annotations

"""
Audio-Visual fusion expert: MulT-lite (two directional cross-modal streams) + pooling.

Pipeline:
  1) per-modality projection -> optional modality encoders
  2) stacked cross-attention blocks (A attends V), (V attends A)
  3) optional stream refinement encoder
  4) attentive stats pooling on each stream -> concat -> classifier

forward(batch) -> logits [B,C]
forward_with_extras(batch) -> (logits, pooled, attn_cat)
  - pooled: fused representation [B, 4H] (concat of two pooled stats vectors: 2H+2H)
  - attn_cat: concatenated pooling attention [B, Ta_max + Tv_max]
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.ser.experts.registry import register_expert


def _key_padding(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
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
        pooled = torch.cat([mu, std], dim=-1)  # [B,2H]
        return pooled, attn


def _pad_attn(attn: torch.Tensor, T_max: int) -> torch.Tensor:
    B, T = attn.shape
    if T == T_max:
        return attn
    out = attn.new_zeros((B, T_max))
    out[:, :T] = attn
    return out


class CrossAttnBlock(nn.Module):
    """
    One cross-attention + FFN block:
      y = LN(y + MHA(query=y, key=context, value=context))
      y = LN(y + FFN(y))
    """

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self,
        query_tokens: torch.Tensor,              # [B,Tq,H]
        context_tokens: torch.Tensor,            # [B,Tc,H]
        context_mask: Optional[torch.Tensor],    # [B,Tc] 1=valid
        query_mask: Optional[torch.Tensor],      # [B,Tq] 1=valid
    ) -> torch.Tensor:
        out, _ = self.mha(
            query=query_tokens,
            key=context_tokens,
            value=context_tokens,
            key_padding_mask=_key_padding(context_mask),
            need_weights=False,
        )
        y = self.ln1(query_tokens + self.drop(out))
        y = self.ln2(y + self.drop(self.ffn(y)))
        if query_mask is not None:
            y = y.masked_fill(query_mask.unsqueeze(-1) == 0, 0.0)
        return y


@register_expert("fusion_av_multlite")
class FusionAVMulTLiteExpert(nn.Module):
    """
    Config keys (expert):

    Required:
      - num_classes
      - audio_input_dim
      - video_input_dim

    Recommended:
      - hidden_dim (256)
      - nhead (4)
      - ffn_dim (1024)
      - dropout (0.4)
      - use_pos_enc (true)

      - num_layers_audio (1)   # optional modality encoders
      - num_layers_video (1)
      - num_layers_xattn (2)   # cross-attn blocks per direction
      - num_layers_refine (1)  # optional refinement per stream
      - attn_hidden (128)
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        self.num_classes = int(cfg["num_classes"])
        self.da = int(cfg["audio_input_dim"])
        self.dv = int(cfg["video_input_dim"])

        self.h = int(cfg.get("hidden_dim", 256))
        self.nhead = int(cfg.get("nhead", 4))
        self.ffn_dim = int(cfg.get("ffn_dim", self.h * 4))
        self.dropout = float(cfg.get("dropout", 0.4))

        self.use_pos = bool(cfg.get("use_pos_enc", True))

        self.la = int(cfg.get("num_layers_audio", 1))
        self.lv = int(cfg.get("num_layers_video", 1))
        self.lx = int(cfg.get("num_layers_xattn", 2))
        self.lr = int(cfg.get("num_layers_refine", 1))

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

        self.pos = SinPosEnc(self.h, max_len=int(cfg.get("pos_max_len", 4000))) if self.use_pos else None

        # cross-attn stacks
        self.a2v_blocks = nn.ModuleList([CrossAttnBlock(self.h, self.nhead, self.ffn_dim, self.dropout) for _ in range(max(1, self.lx))])
        self.v2a_blocks = nn.ModuleList([CrossAttnBlock(self.h, self.nhead, self.ffn_dim, self.dropout) for _ in range(max(1, self.lx))])

        # optional refine encoders on each stream
        self.refine_a = _make_encoder(self.lr)
        self.refine_v = _make_encoder(self.lr)

        attn_hidden = int(cfg.get("attn_hidden", 128))
        self.pool_a = AttnStatsPooling(self.h, attn_hidden=attn_hidden, dropout=self.dropout)
        self.pool_v = AttnStatsPooling(self.h, attn_hidden=attn_hidden, dropout=self.dropout)

        # pooled_a=[B,2H], pooled_v=[B,2H] -> [B,4H]
        self.cls = nn.Sequential(
            nn.LayerNorm(self.h * 4),
            nn.Dropout(self.dropout),
            nn.Linear(self.h * 4, int(cfg.get("mlp_dim", self.h * 2))),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(cfg.get("mlp_dim", self.h * 2)), self.num_classes),
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
        xa = batch["x_audio"]
        ma = batch.get("x_audio_mask", None)
        xv = batch["x_video"]
        mv = batch.get("x_video_mask", None)

        ha = self._encode_mod(xa, ma, self.proj_a, self.enc_a)  # [B,Ta,H]
        hv = self._encode_mod(xv, mv, self.proj_v, self.enc_v)  # [B,Tv,H]

        # A attends V (query=audio, context=video)
        ha2v = ha
        for blk in self.a2v_blocks:
            ha2v = blk(query_tokens=ha2v, context_tokens=hv, context_mask=mv, query_mask=ma)

        # V attends A (query=video, context=audio)
        hv2a = hv
        for blk in self.v2a_blocks:
            hv2a = blk(query_tokens=hv2a, context_tokens=ha, context_mask=ma, query_mask=mv)

        # optional refinement
        if self.refine_a is not None:
            ha2v = self.refine_a(ha2v, src_key_padding_mask=_key_padding(ma))
        if self.refine_v is not None:
            hv2a = self.refine_v(hv2a, src_key_padding_mask=_key_padding(mv))

        pa, attn_a = self.pool_a(ha2v, ma)  # [B,2H], [B,Ta]
        pv, attn_v = self.pool_v(hv2a, mv)  # [B,2H], [B,Tv]

        pooled = torch.cat([pa, pv], dim=-1)  # [B,4H]
        logits = self.cls(pooled)

        Ta_max = xa.shape[1]
        Tv_max = xv.shape[1]
        attn_cat = torch.cat([_pad_attn(attn_a, Ta_max), _pad_attn(attn_v, Tv_max)], dim=1)

        self.last_pooled = pooled
        self.last_attn = attn_cat
        return logits, pooled, attn_cat

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        logits, _, _ = self.forward_with_extras(batch)
        return logits
