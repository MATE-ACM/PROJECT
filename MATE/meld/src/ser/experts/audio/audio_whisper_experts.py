from __future__ import annotations

"""
Whisper audio expert (frame-level npy features).

expert.type = "audio_whisper_experts""

Batch keys (from AudioNpyDataset/collate_audio_npy):
  - x_audio:      [B, T, D]
  - x_audio_mask: [B, T]  (1=valid, 0=pad)
  - y:            [B]

This expert supports plug-in architectures via YAML:
  - encoder: "cnn" | "transformer"
  - pool:    "attn_stats" | "attn_mean"

So you can keep data pipeline unchanged and only swap YAMLs.
"""

from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from src.ser.experts.registry import register_expert

# -------------------------
# Pooling variants
# -------------------------
class AttentionMeanPooling(nn.Module):
    """Weighted mean pooling: [B,T,H] -> [B,H] with attention weights [B,T]."""
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.net(x).squeeze(-1)  # [B,T]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=1)  # [B,T]
        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)  # [B,H]
        return pooled, attn

class AttnStatsPooling(nn.Module):
    """Attentive statistics pooling: weighted mean + std -> [B, 2H]."""
    def __init__(self, hidden_dim: int, dropout: float, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.net(x).squeeze(-1)  # [B,T]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=1)  # [B,T]

        w = attn.unsqueeze(-1)  # [B,T,1]
        mu = torch.sum(w * x, dim=1)                 # [B,H]
        ex2 = torch.sum(w * (x * x), dim=1)          # [B,H]
        var = (ex2 - mu * mu).clamp_min(0.0)
        std = torch.sqrt(var + self.eps)             # [B,H]
        pooled = torch.cat([mu, std], dim=-1)        # [B,2H]
        return pooled, attn

# -------------------------
# Encoder variants
# -------------------------
class ResidualConvBlock(nn.Module):
    """Conv1d residual block over time. x: [B,T,H] -> [B,T,H]."""
    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        k = int(kernel_size)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k // 2)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2))
        y = self.bn(y)
        y = self.act(y)
        y = self.drop(y)
        y = y.transpose(1, 2)
        y = x + y
        if mask is not None:
            y = y.masked_fill(mask.unsqueeze(-1) == 0, 0.0)
        return y

class TransformerEncoder(nn.Module):
    """Transformer encoder over time. x: [B,T,H] -> [B,T,H]."""
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, dropout: float, max_len: int = 6000):
        super().__init__()
        self.max_len = int(max_len)
        self.pos = nn.Parameter(torch.randn(1, self.max_len, hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=hidden_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=int(num_layers))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, H = x.shape
        if T > self.max_len:
            # hard cap (or you can set max_len larger)
            x = x[:, : self.max_len, :]
            if mask is not None:
                mask = mask[:, : self.max_len]
            T = self.max_len

        x = x + self.pos[:, :T, :]
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)  # True=pad
        h = self.enc(x, src_key_padding_mask=key_padding_mask)
        if mask is not None:
            h = h.masked_fill(mask.unsqueeze(-1) == 0, 0.0)
        return h

# -------------------------
# Expert
# -------------------------
@register_expert("audio_whisper_experts")
class AudioWhisperNpyExpert(nn.Module):
    """
    YAML keys (expert):
      type: audio_whisper_npy
      num_classes: 6
      input_dim: 1280
      hidden_dim: 256
      num_layers: 3
      dropout: 0.25

      # arch switches (THIS is what you vary across YAMLs)
      encoder: cnn | transformer
      pool: attn_stats | attn_mean

      # cnn-only
      kernel_size: 3

      # transformer-only
      num_heads: 4
      max_pos_len: 6000

      # optional frozen-feature augmentation
      time_mask_prob: 0.5
      time_mask_max_width: 20
      time_mask_num: 2
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        self.num_classes = int(cfg["num_classes"])
        self.input_dim = int(cfg["input_dim"])
        self.hidden_dim = int(cfg.get("hidden_dim", 256))
        self.num_layers = int(cfg.get("num_layers", 3))
        self.dropout = float(cfg.get("dropout", 0.25))

        self.encoder_type = str(cfg.get("encoder", "cnn")).lower()
        self.pool_type = str(cfg.get("pool", "attn_stats")).lower()

        # cnn-only
        self.kernel_size = int(cfg.get("kernel_size", 3))

        # transformer-only
        self.num_heads = int(cfg.get("num_heads", 4))
        self.max_pos_len = int(cfg.get("max_pos_len", 6000))

        # time masking over projected features (works for both encoders)
        self.time_mask_prob = float(cfg.get("time_mask_prob", 0.0))
        self.time_mask_max_width = int(cfg.get("time_mask_max_width", 20))
        self.time_mask_num = int(cfg.get("time_mask_num", 2))

        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.GELU(),
        )

        # build encoder
        if self.encoder_type == "cnn":
            blocks = []
            for i in range(self.num_layers):
                k = self.kernel_size + 2 * i
                blocks.append(ResidualConvBlock(self.hidden_dim, kernel_size=k, dropout=self.dropout))
            self.encoder = nn.ModuleList(blocks)
        elif self.encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dropout=self.dropout,
                max_len=self.max_pos_len,
            )
        else:
            raise ValueError(f"Unknown encoder: {self.encoder_type}")

        # build pooling + classifier
        if self.pool_type == "attn_stats":
            self.pool = AttnStatsPooling(self.hidden_dim, dropout=self.dropout)
            cls_in = self.hidden_dim * 2
        elif self.pool_type == "attn_mean":
            self.pool = AttentionMeanPooling(self.hidden_dim, dropout=self.dropout)
            cls_in = self.hidden_dim
        else:
            raise ValueError(f"Unknown pool: {self.pool_type}")

        self.classifier = nn.Sequential(
            nn.LayerNorm(cls_in),
            nn.Dropout(self.dropout),
            nn.Linear(cls_in, self.num_classes),
        )

        # for analyze_expert.py
        self.last_pooled: Optional[torch.Tensor] = None
        self.last_attn: Optional[torch.Tensor] = None

    def _apply_time_mask(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.training:
            return x
        if self.time_mask_prob <= 0 or self.time_mask_max_width <= 0 or self.time_mask_num <= 0:
            return x

        B, T, _ = x.shape
        if mask is None:
            mask = x.new_ones((B, T), dtype=torch.long)

        for b in range(B):
            if torch.rand((), device=x.device).item() > self.time_mask_prob:
                continue
            valid = int(mask[b].sum().item())
            if valid <= 1:
                continue
            for _ in range(self.time_mask_num):
                w = int(torch.randint(1, max(2, self.time_mask_max_width + 1), (1,), device=x.device).item())
                w = min(w, valid)
                s = int(torch.randint(0, max(1, valid - w + 1), (1,), device=x.device).item())
                x[b, s:s + w, :] = 0.0
        return x

    def forward_with_extras(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["x_audio"]
        mask = batch.get("x_audio_mask", None)

        x = self.proj(x)
        x = self._apply_time_mask(x, mask)

        if self.encoder_type == "cnn":
            h = x
            for blk in self.encoder:  # type: ignore
                h = blk(h, mask=mask)
        else:
            h = self.encoder(x, mask=mask)  # type: ignore

        pooled, attn = self.pool(h, mask=mask)
        logits = self.classifier(pooled)

        self.last_pooled = pooled
        self.last_attn = attn
        return logits, pooled, attn

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        logits, _, _ = self.forward_with_extras(batch)
        return logits
