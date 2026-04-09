# src/ser/experts/video_experts_universal.py

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ser.experts.registry import register_expert


class AttnMeanPooling(nn.Module):
    def __init__(self, hidden_dim: int, attn_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,T,H], mask: [B,T] 1=valid
        s = self.net(x).squeeze(-1)  # [B,T]
        if mask is not None:
            s = s.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(s, dim=1)  # [B,T]
        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)  # [B,H]
        return pooled, attn


class AttnStatsPooling(nn.Module):
    def __init__(self, hidden_dim: int, attn_hidden: int = 128, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # mean+std -> [B,2H]
        s = self.net(x).squeeze(-1)  # [B,T]
        if mask is not None:
            s = s.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(s, dim=1)  # [B,T]
        w = attn.unsqueeze(-1)          # [B,T,1]
        mu = torch.sum(w * x, dim=1)    # [B,H]
        ex2 = torch.sum(w * (x * x), dim=1)
        var = (ex2 - mu * mu).clamp_min(0.0)
        std = torch.sqrt(var + self.eps)
        pooled = torch.cat([mu, std], dim=-1)  # [B,2H]
        return pooled, attn


class ResidualTCNGN(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float, num_groups: int = 8):
        super().__init__()
        k = int(kernel_size)
        pad = (k - 1) // 2
        g = min(num_groups, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=pad)
        self.gn1 = nn.GroupNorm(g, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=pad)
        self.gn2 = nn.GroupNorm(g, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [B,T,H]
        r = x
        y = x.transpose(1, 2)          # [B,H,T]
        y = self.conv1(y)
        y = self.gn1(y)
        y = F.gelu(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.gn2(y)
        y = F.gelu(y)
        y = self.drop(y)
        y = y.transpose(1, 2)          # [B,T,H]
        y = r + y
        if mask is not None:
            y = y.masked_fill(mask.unsqueeze(-1) == 0, 0.0)
        return y


class SinPosEnc(nn.Module):
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


@register_expert("video_experts")
class VideoExpertsUniversal(nn.Module):
    """
    mode:
      - seq:  in_proj -> (tcn_gn/transformer) -> pool -> cls
      - linear_attn: attention pool on raw features -> linear cls (最可解释，适合OpenFace)
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.num_classes = int(cfg["num_classes"])
        self.input_dim = int(cfg["input_dim"])

        self.mode = str(cfg.get("mode", "seq"))  # seq | linear_attn
        self.encoder = str(cfg.get("encoder", "tcn_gn"))  # tcn_gn | transformer
        self.pool_type = str(cfg.get("pool", "attn_stats"))  # attn_stats | attn_mean

        self.hidden_dim = int(cfg.get("hidden_dim", 256))
        self.num_layers = int(cfg.get("num_layers", 3))
        self.dropout = float(cfg.get("dropout", 0.3))
        self.kernel_size = int(cfg.get("kernel_size", 3))

        # pooling
        if self.pool_type == "attn_stats":
            self.pool = AttnStatsPooling(self.hidden_dim if self.mode == "seq" else self.input_dim,
                                         attn_hidden=int(cfg.get("attn_hidden", 128)))
            pooled_dim = 2 * (self.hidden_dim if self.mode == "seq" else self.input_dim)
        elif self.pool_type == "attn_mean":
            self.pool = AttnMeanPooling(self.hidden_dim if self.mode == "seq" else self.input_dim,
                                        attn_hidden=int(cfg.get("attn_hidden", 128)))
            pooled_dim = (self.hidden_dim if self.mode == "seq" else self.input_dim)
        else:
            raise KeyError(f"Unknown pool={self.pool_type}")

        if self.mode == "linear_attn":
            # 可解释：分类器直接吃 pooled 原始特征（例如OpenFace AUs）
            self.cls = nn.Linear(pooled_dim, self.num_classes)
            return

        # seq mode: in_proj + encoder
        self.in_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.in_ln = nn.LayerNorm(self.hidden_dim)
        self.in_drop = nn.Dropout(self.dropout)

        if self.encoder == "tcn_gn":
            self.blocks = nn.ModuleList([
                ResidualTCNGN(self.hidden_dim, kernel_size=self.kernel_size, dropout=self.dropout)
                for _ in range(self.num_layers)
            ])
            self.seq = None

        elif self.encoder == "transformer":
            nhead = int(cfg.get("nhead", 4))
            ff = int(cfg.get("ffn_dim", self.hidden_dim * 4))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim, nhead=nhead,
                dim_feedforward=ff, dropout=self.dropout,
                batch_first=True, activation="gelu",
            )
            self.pos = SinPosEnc(self.hidden_dim)
            self.seq = nn.TransformerEncoder(enc_layer, num_layers=max(1, self.num_layers))
            self.blocks = None
        else:
            raise KeyError(f"Unknown encoder={self.encoder}")

        self.cls = nn.Linear(pooled_dim, self.num_classes)

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.in_ln(h)
        h = F.gelu(h)
        h = self.in_drop(h)

        if self.encoder == "tcn_gn":
            for b in self.blocks:
                h = b(h, mask)
            return h

        # transformer
        h = self.pos(h)
        key_padding = (mask == 0) if mask is not None else None
        return self.seq(h, src_key_padding_mask=key_padding)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        x = batch["x_video"]
        m = batch.get("x_video_mask", None)

        if self.mode == "linear_attn":
            pooled, _ = self.pool(x, m)
            return self.cls(pooled)

        h = self.encode(x, m)
        pooled, _ = self.pool(h, m)
        return self.cls(pooled)

    def forward_with_extras(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["x_video"]
        m = batch.get("x_video_mask", None)

        if self.mode == "linear_attn":
            pooled, attn = self.pool(x, m)
            logits = self.cls(pooled)
            return logits, pooled, attn

        h = self.encode(x, m)
        pooled, attn = self.pool(h, m)
        logits = self.cls(pooled)
        return logits, pooled, attn
