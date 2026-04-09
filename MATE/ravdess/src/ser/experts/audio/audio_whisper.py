from __future__ import annotations

"""
Whisper audio expert (frame-level npy features).

expert.type = "audio_whisper_npy"

Input batch keys (from AudioNpyDataset/collate_audio_npy):
  - x_audio:      [B, T, D]
  - x_audio_mask: [B, T]  (1=valid, 0=pad)
  - y:            [B]
Outputs:
  - forward(batch) -> logits [B, C]
  - forward_with_extras(batch) -> (logits, pooled, attn) for analyze_expert.py
"""

from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from src.ser.experts.registry  import register_expert


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


@register_expert("audio_whisper_npy")
class AudioWhisperNpyExpert(nn.Module):
    """
    YAML (expert) keys:
      type: audio_whisper_npy
      num_classes: 6
      input_dim: 1280
      hidden_dim: 256
      num_layers: 3
      dropout: 0.25
      kernel_size: 3

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
        self.kernel_size = int(cfg.get("kernel_size", 3))

        self.time_mask_prob = float(cfg.get("time_mask_prob", 0.0))
        self.time_mask_max_width = int(cfg.get("time_mask_max_width", 20))
        self.time_mask_num = int(cfg.get("time_mask_num", 2))

        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.GELU(),
        )

        blocks = []
        for i in range(self.num_layers):
            k = self.kernel_size + 2 * i
            blocks.append(ResidualConvBlock(self.hidden_dim, kernel_size=k, dropout=self.dropout))
        self.cnn = nn.ModuleList(blocks)

        self.pool = AttnStatsPooling(self.hidden_dim, dropout=self.dropout)
        cls_in = self.hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.LayerNorm(cls_in),
            nn.Dropout(self.dropout),
            nn.Linear(cls_in, self.num_classes),
        )

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

        h = x
        for blk in self.cnn:
            h = blk(h, mask=mask)

        pooled, attn = self.pool(h, mask=mask)
        logits = self.classifier(pooled)

        self.last_pooled = pooled
        self.last_attn = attn
        return logits, pooled, attn

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        logits, _, _ = self.forward_with_extras(batch)
        return logits
