from __future__ import annotations

"""
Video experts (npz/npy frame-level features).

expert.type = "video_experts"

Batch keys (from VideoNpyDataset/collate_video_npy):
  - x_video:      [B, T, D]
  - x_video_mask: [B, T]  (1=valid, 0=pad)
  - y:            [B]

Config keys (expert):
  - type: video_experts
  - input_dim, num_classes
  - mode: "seq"  (保留字段；目前统一按 seq 处理)
  - encoder: "none" | "gru" | "tcn_gn" | "tcn_bn" | "transformer"
  - pool: "attn_stats" | "attn_mean"
  - hidden_dim, num_layers, dropout
  - (transformer) nhead, ffn_dim
  - (tcn) kernel_size
  - (gru) bidirectional (bool)

Why this file:
- 用同一个 expert.type + YAML 切换不同“时序头”，方便做 ablation。
- 输出 forward_with_extras -> (logits, pooled, attn) 便于你后续融合/诊断。
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.ser.experts.registry import register_expert


# -------------------------
# Pooling
# -------------------------
class AttentiveMeanPooling(nn.Module):
    def __init__(self, dim: int, attn_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,T,H], mask: [B,T]
        s = self.net(x).squeeze(-1)  # [B,T]
        if mask is not None:
            s = s.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(s, dim=1)  # [B,T]
        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)  # [B,H]
        return pooled, attn


class AttentiveStatsPooling(nn.Module):
    def __init__(self, dim: int, attn_hidden: int = 128, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 输出 [B, 2H] = concat(mean, std)
        s = self.net(x).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(s, dim=1)
        w = attn.unsqueeze(-1)  # [B,T,1]

        mu = torch.sum(w * x, dim=1)
        ex2 = torch.sum(w * (x * x), dim=1)
        var = (ex2 - mu * mu).clamp_min(0.0)
        std = torch.sqrt(var + self.eps)

        pooled = torch.cat([mu, std], dim=-1)  # [B,2H]
        return pooled, attn


# -------------------------
# Encoders
# -------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(x.dtype)


class ResidualTCN_BN(nn.Module):
    """TCN block with BatchNorm (你原来那种)"""
    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        k = int(kernel_size)
        pad = (k - 1) // 2
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=pad)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=pad)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        r = x
        y = x.transpose(1, 2)      # [B,H,T]
        y = self.conv1(y)
        y = self.bn1(y)
        y = F.gelu(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.gelu(y)
        y = self.drop(y)
        y = y.transpose(1, 2)      # [B,T,H]
        y = r + y
        if mask is not None:
            y = y.masked_fill(mask.unsqueeze(-1) == 0, 0.0)
        return y


class ResidualTCN_GN(nn.Module):
    """OpenFace)"""
    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float, num_groups: int = 8):
        super().__init__()
        k = int(kernel_size)
        pad = (k - 1) // 2
        g = min(num_groups, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=pad)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=pad)
        self.gn1 = nn.GroupNorm(g, hidden_dim)
        self.gn2 = nn.GroupNorm(g, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        r = x
        y = x.transpose(1, 2)
        y = self.conv1(y)
        y = self.gn1(y)
        y = F.gelu(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.gn2(y)
        y = F.gelu(y)
        y = self.drop(y)
        y = y.transpose(1, 2)
        y = r + y
        if mask is not None:
            y = y.masked_fill(mask.unsqueeze(-1) == 0, 0.0)
        return y


# -------------------------
# Main Expert
# -------------------------
@register_expert("video_hsemotion_experts")
class VideoExperts(nn.Module):
    """
    你要的 3 个轻量时序头：
      1) encoder=none + pool=attn_stats  -> ASP（最轻）
      2) encoder=gru  + pool=attn_stats  -> BiGRU+Attn（经典强基线）
      3) encoder=tcn_gn + pool=attn_stats -> TCN-GN（又快又强）

    也保留你现有：
      - encoder=transformer
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        self.input_dim = int(cfg["input_dim"])
        self.num_classes = int(cfg["num_classes"])

        self.mode = str(cfg.get("mode", "seq"))
        self.encoder_type = str(cfg.get("encoder", "none"))
        self.pool_type = str(cfg.get("pool", "attn_stats"))

        self.hidden_dim = int(cfg.get("hidden_dim", 256))
        self.num_layers = int(cfg.get("num_layers", 2))
        self.dropout = float(cfg.get("dropout", 0.3))
        self.kernel_size = int(cfg.get("kernel_size", 3))

        # input projection (统一把 D -> hidden_dim，保证不同 encoder 头可比较)
        self.in_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.in_ln = nn.LayerNorm(self.hidden_dim)
        self.in_drop = nn.Dropout(self.dropout)

        # build encoder
        self.pos = None
        self.tr = None
        self.rnn = None
        self.blocks = None

        if self.encoder_type == "none":
            # 不做时序编码，直接 pooling
            pass

        elif self.encoder_type == "gru":
            bidir = bool(cfg.get("bidirectional", True))
            # 让输出维度恒为 hidden_dim：双向时每向 hidden_dim//2
            rnn_hidden = self.hidden_dim // 2 if bidir else self.hidden_dim
            if bidir and self.hidden_dim % 2 != 0:
                raise ValueError("For bidirectional GRU, please use even hidden_dim.")
            self.rnn = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=rnn_hidden,
                num_layers=max(1, self.num_layers),
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0.0,
                bidirectional=bidir,
            )

        elif self.encoder_type in ("tcn_gn", "tcn_bn"):
            Block = ResidualTCN_GN if self.encoder_type == "tcn_gn" else ResidualTCN_BN
            self.blocks = nn.ModuleList([
                Block(self.hidden_dim, kernel_size=self.kernel_size, dropout=self.dropout)
                for _ in range(max(1, self.num_layers))
            ])

        elif self.encoder_type == "transformer":
            nhead = int(cfg.get("nhead", 4))
            ffn_dim = int(cfg.get("ffn_dim", self.hidden_dim * 4))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=nhead,
                dim_feedforward=ffn_dim,
                dropout=self.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.pos = SinusoidalPositionalEncoding(self.hidden_dim)
            self.tr = nn.TransformerEncoder(enc_layer, num_layers=max(1, self.num_layers))

        else:
            raise KeyError(f"Unknown encoder={self.encoder_type}")

        # build pooling
        attn_hidden = int(cfg.get("attn_hidden", 128))
        if self.pool_type == "attn_stats":
            self.pool = AttentiveStatsPooling(self.hidden_dim, attn_hidden=attn_hidden)
            out_dim = 2 * self.hidden_dim
        elif self.pool_type == "attn_mean":
            self.pool = AttentiveMeanPooling(self.hidden_dim, attn_hidden=attn_hidden)
            out_dim = self.hidden_dim
        else:
            raise KeyError(f"Unknown pool={self.pool_type}")

        self.cls = nn.Linear(out_dim, self.num_classes)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.in_ln(h)
        h = F.gelu(h)
        h = self.in_drop(h)
        return h

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [B,T,D] -> h: [B,T,H]
        h = self._project(x)

        if self.encoder_type == "none":
            return h

        if self.encoder_type == "gru":
            assert self.rnn is not None
            if mask is None:
                out, _ = self.rnn(h)
                return out

            lengths = mask.sum(dim=1).clamp(min=1).to(torch.long).cpu()
            lengths_sorted, idx = torch.sort(lengths, descending=True)
            h_sorted = h.index_select(0, idx.to(h.device))

            packed = pack_padded_sequence(h_sorted, lengths_sorted, batch_first=True, enforce_sorted=True)
            packed_out, _ = self.rnn(packed)
            out_sorted, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=h.size(1))

            # unsort
            inv = torch.empty_like(idx)
            inv[idx] = torch.arange(idx.size(0), device=idx.device)
            out = out_sorted.index_select(0, inv.to(out_sorted.device))

            # 把 pad 位置清零，避免 attention 误吃
            out = out.masked_fill(mask.unsqueeze(-1) == 0, 0.0)
            return out

        if self.encoder_type in ("tcn_gn", "tcn_bn"):
            assert self.blocks is not None
            for b in self.blocks:
                h = b(h, mask)
            return h

        # transformer
        assert self.tr is not None and self.pos is not None
        h = self.pos(h)
        key_padding = (mask == 0) if mask is not None else None
        out = self.tr(h, src_key_padding_mask=key_padding)
        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(-1) == 0, 0.0)
        return out

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        x = batch["x_video"]
        m = batch.get("x_video_mask", None)
        h = self.encode(x, m)
        pooled, _attn = self.pool(h, m)
        logits = self.cls(pooled)
        return logits

    def forward_with_extras(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["x_video"]
        m = batch.get("x_video_mask", None)
        h = self.encode(x, m)
        pooled, attn = self.pool(h, m)
        logits = self.cls(pooled)
        return logits, pooled, attn
