from __future__ import annotations

"""src.ser.experts.txt.txt_experts

Text experts for frozen text features (e.g., RoBERTa / BERT embeddings saved as .npy).

expert.type = "txt_experts" (prefix "txt_" so src.ser.data.dataloaders routes correctly)

Expected batch keys (from TxtNpyDataset/collate_txt_npy):
  - x_txt:      [B, T, D]
  - x_txt_mask: [B, T]  (1=valid, 0=pad)
  - y:          [B]

Design goal:
- Keep the same "YAML switch" style as video_hsemotion_experts / audio_whisper_experts.
- Provide very typical SER heads that work well on frozen embeddings:
  * encoder=none + attentive pooling  (ASP baseline)
  * encoder=gru/lstm (BiRNN) + attentive pooling
  * encoder=tcn_gn (TCN-GN) + attentive pooling
  * encoder=transformer + attentive pooling

Notes:
- If your saved feature is utterance-level [D], the dataset will auto-reshape it to [1, D],
  so all models still work (T=1).
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
        attn = torch.softmax(s, dim=1)
        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)
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
        # returns pooled: [B,2H] = concat(mean,std)
        s = self.net(x).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(s, dim=1)
        w = attn.unsqueeze(-1)  # [B,T,1]

        mu = torch.sum(w * x, dim=1)
        ex2 = torch.sum(w * (x * x), dim=1)
        var = (ex2 - mu * mu).clamp_min(0.0)
        std = torch.sqrt(var + self.eps)
        pooled = torch.cat([mu, std], dim=-1)
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
        y = x.transpose(1, 2)  # [B,H,T]
        y = self.conv1(y)
        y = self.bn1(y)
        y = F.gelu(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.gelu(y)
        y = self.drop(y)
        y = y.transpose(1, 2)
        y = r + y
        if mask is not None:
            y = y.masked_fill(mask.unsqueeze(-1) == 0, 0.0)
        return y

class ResidualTCN_GN(nn.Module):
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
@register_expert("txt_experts")
class TxtExperts(nn.Module):
    """Universal text expert for frozen embeddings.

    YAML keys (expert):
      type: txt_experts
      feat_root: ...
      input_dim: 1024
      num_classes: 4  # will be overridden by scripts/train.py according to dataset labels

      # switches
      encoder: none | gru | lstm | tcn_gn | tcn_bn | transformer
      pool:   attn_stats | attn_mean
      head:   linear | mlp

      hidden_dim: 256
      num_layers: 2
      dropout: 0.35

      # RNN
      bidirectional: true

      # TCN
      kernel_size: 3

      # Transformer
      nhead: 4
      ffn_dim: 1024

      # pooling
      attn_hidden: 128

      # augmentation (optional)
      time_mask_prob: 0.0
      time_mask_max_width: 20
      time_mask_num: 2
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        self.input_dim = int(cfg["input_dim"])
        self.num_classes = int(cfg["num_classes"])

        self.encoder_type = str(cfg.get("encoder", "none")).lower()
        self.pool_type = str(cfg.get("pool", "attn_stats")).lower()
        self.head_type = str(cfg.get("head", "linear")).lower()

        self.hidden_dim = int(cfg.get("hidden_dim", 256))
        self.num_layers = int(cfg.get("num_layers", 2))
        self.dropout = float(cfg.get("dropout", 0.35))

        # rnn
        self.bidirectional = bool(cfg.get("bidirectional", True))

        # tcn
        self.kernel_size = int(cfg.get("kernel_size", 3))

        # transformer
        self.nhead = int(cfg.get("nhead", 4))
        self.ffn_dim = int(cfg.get("ffn_dim", self.hidden_dim * 4))

        # pooling
        self.attn_hidden = int(cfg.get("attn_hidden", 128))

        # time masking (SpecAugment-style) over projected features
        self.time_mask_prob = float(cfg.get("time_mask_prob", 0.0))
        self.time_mask_max_width = int(cfg.get("time_mask_max_width", 20))
        self.time_mask_num = int(cfg.get("time_mask_num", 2))

        # input projection
        self.in_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.in_ln = nn.LayerNorm(self.hidden_dim)
        self.in_drop = nn.Dropout(self.dropout)

        # encoder modules
        self.pos = None
        self.tr = None
        self.rnn = None
        self.blocks = None

        if self.encoder_type == "none":
            pass

        elif self.encoder_type in {"gru", "lstm"}:
            bidir = self.bidirectional
            # keep output dim = hidden_dim (if bidir, each direction uses hidden_dim//2)
            rnn_hidden = self.hidden_dim // 2 if bidir else self.hidden_dim
            if bidir and self.hidden_dim % 2 != 0:
                raise ValueError("For bidirectional RNN, please use even hidden_dim.")

            rnn_cls = nn.GRU if self.encoder_type == "gru" else nn.LSTM
            self.rnn = rnn_cls(
                input_size=self.hidden_dim,
                hidden_size=rnn_hidden,
                num_layers=max(1, self.num_layers),
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0.0,
                bidirectional=bidir,
            )

        elif self.encoder_type in {"tcn_gn", "tcn_bn"}:
            Block = ResidualTCN_GN if self.encoder_type == "tcn_gn" else ResidualTCN_BN
            self.blocks = nn.ModuleList(
                [
                    Block(self.hidden_dim, kernel_size=self.kernel_size, dropout=self.dropout)
                    for _ in range(max(1, self.num_layers))
                ]
            )

        elif self.encoder_type == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.nhead,
                dim_feedforward=self.ffn_dim,
                dropout=self.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.pos = SinusoidalPositionalEncoding(self.hidden_dim)
            self.tr = nn.TransformerEncoder(enc_layer, num_layers=max(1, self.num_layers))

        else:
            raise KeyError(f"Unknown encoder={self.encoder_type}")

        # pooling
        if self.pool_type == "attn_stats":
            self.pool = AttentiveStatsPooling(self.hidden_dim, attn_hidden=self.attn_hidden)
            pooled_dim = 2 * self.hidden_dim
        elif self.pool_type == "attn_mean":
            self.pool = AttentiveMeanPooling(self.hidden_dim, attn_hidden=self.attn_hidden)
            pooled_dim = self.hidden_dim
        else:
            raise KeyError(f"Unknown pool={self.pool_type}")

        # classifier head
        if self.head_type == "linear":
            self.cls = nn.Sequential(
                nn.LayerNorm(pooled_dim),
                nn.Dropout(self.dropout),
                nn.Linear(pooled_dim, self.num_classes),
            )
        elif self.head_type == "mlp":
            head_hidden = int(cfg.get("head_hidden", pooled_dim))
            self.cls = nn.Sequential(
                nn.LayerNorm(pooled_dim),
                nn.Linear(pooled_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(head_hidden, self.num_classes),
            )
        else:
            raise KeyError(f"Unknown head={self.head_type}")

        # for analysis scripts
        self.last_pooled: Optional[torch.Tensor] = None
        self.last_attn: Optional[torch.Tensor] = None

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.in_ln(h)
        h = F.gelu(h)
        h = self.in_drop(h)
        return h

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
                x[b, s : s + w, :] = 0.0
        return x

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [B,T,D] -> h: [B,T,H]
        h = self._project(x)
        h = self._apply_time_mask(h, mask)

        if self.encoder_type == "none":
            return h

        if self.encoder_type in {"gru", "lstm"}:
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

            out = out.masked_fill(mask.unsqueeze(-1) == 0, 0.0)
            return out

        if self.encoder_type in {"tcn_gn", "tcn_bn"}:
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
        x = batch["x_txt"]
        m = batch.get("x_txt_mask", None)
        h = self.encode(x, m)
        pooled, _attn = self.pool(h, m)
        logits = self.cls(pooled)
        return logits

    def forward_with_extras(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["x_txt"]
        m = batch.get("x_txt_mask", None)
        h = self.encode(x, m)
        pooled, attn = self.pool(h, m)
        logits = self.cls(pooled)
        self.last_pooled = pooled
        self.last_attn = attn
        return logits, pooled, attn
