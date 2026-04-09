from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import os

import torch
import torch.nn as nn

from src.ser.experts.registry import register_expert


# =========================
# Basic blocks (same style)
# =========================

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


def _key_padding(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return (mask == 0)


# =========================
# Meta CSV table
# =========================

class MetaCSVTable:
    """
    CSV -> uid_to_idx + feats_cpu

    Default:
      - only numeric columns are used as features
      - uid_col is excluded even if numeric
      - NaN/Inf -> 0
    """

    def __init__(
        self,
        csv_path: str,
        uid_col: str = "uid",
        feature_cols: Optional[List[str]] = None,
        drop_cols: Optional[List[str]] = None,
        missing: str = "zeros",  # 'zeros' or 'error'
        dtype: torch.dtype = torch.float32,
    ):
        self.csv_path = csv_path
        self.uid_col = uid_col
        self.missing = missing
        self.dtype = dtype

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"[MetaCSVTable] meta_csv_path not found: {csv_path}")

        # Prefer pandas (fast + robust); fallback to csv module if pandas missing
        try:
            import pandas as pd  # type: ignore
            import numpy as np  # type: ignore

            df = pd.read_csv(csv_path)
            if uid_col not in df.columns:
                raise ValueError(f"[MetaCSVTable] uid_col='{uid_col}' not in CSV columns.")

            if feature_cols is None:
                # safest: numeric only
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                feature_cols = [c for c in numeric_cols if c != uid_col]
            else:
                feature_cols = [c for c in feature_cols if c != uid_col]
                miss = [c for c in feature_cols if c not in df.columns]
                if miss:
                    raise ValueError(f"[MetaCSVTable] feature_cols missing in CSV: {miss[:10]} ...")

            if drop_cols:
                drop = set(drop_cols)
                feature_cols = [c for c in feature_cols if c not in drop]

            keep = [uid_col] + feature_cols
            df = df[keep]

            uids = df[uid_col].astype(str).tolist()
            feats = df[feature_cols].to_numpy()
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

        except Exception:
            # fallback: built-in csv (slower but works)
            import csv
            import numpy as np  # type: ignore

            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None or uid_col not in reader.fieldnames:
                    raise ValueError(f"[MetaCSVTable] uid_col='{uid_col}' not in CSV header.")

                # choose cols
                if feature_cols is None:
                    # try all columns except uid, and parse float; non-float become ignored
                    cand = [c for c in reader.fieldnames if c != uid_col]
                else:
                    cand = [c for c in feature_cols if c != uid_col]

                if drop_cols:
                    drop = set(drop_cols)
                    cand = [c for c in cand if c not in drop]

                uids = []
                rows = []
                usable = None  # decide usable numeric columns by first row
                for row in reader:
                    uids.append(str(row[uid_col]))
                    if usable is None:
                        usable = []
                        for c in cand:
                            try:
                                float(row[c])
                                usable.append(c)
                            except Exception:
                                pass
                    rows.append([float(row[c]) if row[c] not in ("", None) else 0.0 for c in usable])

                feature_cols = usable if usable is not None else []
                feats = np.asarray(rows, dtype="float32")
                feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

        self.feature_cols = feature_cols
        self.dim = int(feats.shape[1]) if feats.ndim == 2 else 0
        self.uid_to_idx: Dict[str, int] = {u: i for i, u in enumerate(uids)}
        self.feats_cpu = torch.from_numpy(feats).to(dtype=self.dtype)  # CPU

    def lookup(self, uids: List[str], device: torch.device) -> torch.Tensor:
        if len(uids) == 0:
            return torch.zeros((0, self.dim), device=device, dtype=self.dtype)

        idxs: List[int] = []
        miss: List[str] = []
        for u in uids:
            ix = self.uid_to_idx.get(u, -1)
            idxs.append(ix)
            if ix < 0:
                miss.append(u)

        if miss and self.missing == "error":
            raise KeyError(f"[MetaCSVTable] missing {len(miss)} uids (first 5): {miss[:5]}")

        out = torch.zeros((len(uids), self.dim), dtype=self.dtype)  # CPU
        good_pos = [i for i, ix in enumerate(idxs) if ix >= 0]
        if good_pos:
            src_idx = torch.tensor([idxs[i] for i in good_pos], dtype=torch.long)
            out[torch.tensor(good_pos, dtype=torch.long)] = self.feats_cpu.index_select(0, src_idx)

        return out.to(device=device, non_blocking=True)


# =========================
# Expert: A + V + Meta
# =========================

@register_expert("fusion_av_pool_mlp_meta")
class FusionAVPoolMLPMetaExpert(nn.Module):
    """
    Late fusion baseline (A + V + Meta):
      audio: proj -> (optional enc) -> attentive stats pooling -> [B,2H]
      video: proj -> (optional enc) -> attentive stats pooling -> [B,2H]
      meta:  CSV lookup by inferred uid -> proj_m -> [B,meta_out]
      concat -> MLP -> logits

    No dataloader changes required:
      - If batch has x_meta: use it
      - Else: infer uids from batch keys (uid/id/path/name/file/...) and lookup CSV
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        self.num_classes = int(cfg["num_classes"])
        self.da = int(cfg["audio_input_dim"])
        self.dv = int(cfg["video_input_dim"])

        self.h = int(cfg.get("hidden_dim", 256))
        self.dropout = float(cfg.get("dropout", 0.3))
        self.use_pos = bool(cfg.get("use_pos_enc", True))

        # ---- meta config ----
        self.use_meta = bool(cfg.get("use_meta", True))
        self.batch_uid_key = str(cfg.get("batch_uid_key", "auto"))  # 'auto' recommended
        self.meta_uid_col = str(cfg.get("meta_uid_col", "uid"))
        self.meta_missing = str(cfg.get("meta_missing", "zeros"))   # 'zeros' | 'error'
        self.uid_match_min_rate = float(cfg.get("uid_match_min_rate", 0.90))

        self.meta_feature_cols = cfg.get("meta_feature_cols", None)  # optional list[str]
        self.meta_drop_cols = cfg.get("meta_drop_cols", None)        # optional list[str]

        self.meta_table: Optional[MetaCSVTable] = None
        self.meta_dim: int = 0
        if self.use_meta:
            meta_csv_path = cfg.get("meta_csv_path", None)
            if meta_csv_path is None:
                raise ValueError("[FusionAVPoolMLPMetaExpert] use_meta=True but cfg['meta_csv_path'] not set.")
            self.meta_table = MetaCSVTable(
                csv_path=str(meta_csv_path),
                uid_col=self.meta_uid_col,
                feature_cols=self.meta_feature_cols,
                drop_cols=self.meta_drop_cols,
                missing=self.meta_missing,
                dtype=torch.float32,
            )
            self.meta_dim = self.meta_table.dim
            if self.meta_dim <= 0:
                raise ValueError(f"[FusionAVPoolMLPMetaExpert] meta_dim invalid (got {self.meta_dim}).")

        # ---- projections ----
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

        # ---- optional encoders (kept cheap) ----
        # Support both key styles:
        # - num_layers_mod (generic)
        # - num_layers_audio / num_layers_video (your YAML)
        nhead = int(cfg.get("nhead", 4))
        ffn_dim = int(cfg.get("ffn_dim", self.h * 4))

        la = int(cfg.get("num_layers_audio", cfg.get("num_layers_mod", 0)))
        lv = int(cfg.get("num_layers_video", cfg.get("num_layers_mod", 0)))

        if la > 0:
            layer_a = nn.TransformerEncoderLayer(
                d_model=self.h,
                nhead=nhead,
                dim_feedforward=ffn_dim,
                dropout=self.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.enc_a = nn.TransformerEncoder(layer_a, num_layers=la)
        else:
            self.enc_a = None

        if lv > 0:
            layer_v = nn.TransformerEncoderLayer(
                d_model=self.h,
                nhead=nhead,
                dim_feedforward=ffn_dim,
                dropout=self.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.enc_v = nn.TransformerEncoder(layer_v, num_layers=lv)
        else:
            self.enc_v = None

        self.pos = SinPosEnc(self.h, max_len=int(cfg.get("pos_max_len", 4000))) if self.use_pos else None

        attn_hidden = int(cfg.get("attn_hidden", 128))
        self.pool_a = AttnStatsPooling(self.h, attn_hidden=attn_hidden, dropout=self.dropout)
        self.pool_v = AttnStatsPooling(self.h, attn_hidden=attn_hidden, dropout=self.dropout)

        # ---- meta projection ----
        meta_out = int(cfg.get("meta_out_dim", self.h * 2)) if self.use_meta else 0
        if self.use_meta:
            self.proj_m = nn.Sequential(
                nn.Linear(self.meta_dim, meta_out),
                nn.LayerNorm(meta_out),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(meta_out, meta_out),
                nn.GELU(),
                nn.Dropout(self.dropout),
            )
        else:
            self.proj_m = None

        # ---- classifier ----
        in_dim = self.h * 4 + meta_out
        mlp_dim = int(cfg.get("mlp_dim", self.h * 2))
        self.cls = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(self.dropout),
            nn.Linear(in_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(mlp_dim, self.num_classes),
        )

        self._uid_infer_logged = False

    # -------------------------
    # uid inference helpers
    # -------------------------

    def _normalize_uid(self, s: Any) -> str:
        s = str(s)
        s = os.path.basename(s)

        # try cfg-defined extensions first
        exts: List[str] = []
        if "audio_ext" in self.cfg:
            exts.append(str(self.cfg["audio_ext"]))
        if "video_ext" in self.cfg:
            exts.append(str(self.cfg["video_ext"]))
        # plus common
        exts += [".npy", ".npz", ".wav", ".mp4", ".avi", ".flac", ".m4a"]

        for ext in exts:
            if ext and s.endswith(ext):
                s = s[: -len(ext)]
        return s

    def _try_make_uid_list(self, v: Any, B: int) -> Optional[List[str]]:
        if v is None:
            return None
        if torch.is_tensor(v):
            if v.numel() != B:
                return None
            return [self._normalize_uid(x.item()) for x in v.detach().cpu().flatten()]
        if isinstance(v, (list, tuple)):
            if len(v) != B:
                return None
            out = []
            for x in v:
                if isinstance(x, (bytes, bytearray)):
                    out.append(self._normalize_uid(x.decode("utf-8")))
                else:
                    out.append(self._normalize_uid(x))
            return out
        try:
            import numpy as np  # type: ignore
            if isinstance(v, np.ndarray):
                if v.size != B:
                    return None
                return [self._normalize_uid(x) for x in v.flatten().tolist()]
        except Exception:
            pass
        return None

    def _infer_uids_from_batch(self, batch: Dict[str, Any], B: int) -> List[str]:
        assert self.meta_table is not None

        keys = list(batch.keys())
        # heuristic candidate ordering
        pri: List[str] = []
        pri += [k for k in keys if "uid" in k.lower()]
        pri += [k for k in keys if any(t in k.lower() for t in ["utt", "seg", "id", "name", "file", "key"])]
        pri += [k for k in keys if "path" in k.lower()]
        pri += [k for k in keys if any(t in k.lower() for t in ["audio", "video"])]  # last resort

        seen = set()
        cand_keys: List[str] = []
        for k in pri:
            if k not in seen:
                seen.add(k)
                cand_keys.append(k)

        scored = []
        for k in cand_keys:
            u = self._try_make_uid_list(batch.get(k, None), B)
            if u is None:
                continue
            hit = sum(1 for x in u if x in self.meta_table.uid_to_idx)
            scored.append((hit / max(B, 1), hit, k, u))

        if not scored:
            raise KeyError(
                f"[Meta] cannot infer uid: batch has no usable id/path field. "
                f"Available keys={list(batch.keys())}"
            )

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_rate, best_hit, best_key, best_uids = scored[0]

        if best_rate < self.uid_match_min_rate:
            top = "; ".join([f"{k}:{r:.2f}({h}/{B})" for r, h, k, _ in scored[:8]])
            raise KeyError(
                f"[Meta] inferred uid key too weak. best={best_key} rate={best_rate:.2f} ({best_hit}/{B}). "
                f"Top candidates: {top}. Set cfg['batch_uid_key'] to the correct key or lower uid_match_min_rate."
            )

        if not self._uid_infer_logged:
            print(f"[Meta] auto uid key='{best_key}' hit_rate={best_rate:.2f} ({best_hit}/{B})")
            self._uid_infer_logged = True

        return best_uids

    # -------------------------
    # core forward
    # -------------------------

    def _encode(self, x: torch.Tensor, mask: Optional[torch.Tensor], proj: nn.Module, enc: Optional[nn.Module]) -> torch.Tensor:
        h = proj(x)
        if self.pos is not None:
            h = self.pos(h)
        if enc is not None:
            h = enc(h, src_key_padding_mask=_key_padding(mask))
        return h

    def _get_x_meta(self, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
        if not self.use_meta:
            raise RuntimeError("[Meta] use_meta=False but _get_x_meta called.")
        if self.meta_table is None:
            raise RuntimeError("[Meta] meta_table is None but use_meta=True")

        # 1) batch provides x_meta
        if "x_meta" in batch and batch["x_meta"] is not None:
            xm = batch["x_meta"]
            if not torch.is_tensor(xm):
                xm = torch.tensor(xm, dtype=torch.float32)
            xm = xm.to(device=device, dtype=torch.float32, non_blocking=True)
            return torch.nan_to_num(xm, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) infer batch size
        if "x_audio" in batch and torch.is_tensor(batch["x_audio"]):
            B = int(batch["x_audio"].shape[0])
        elif "x_video" in batch and torch.is_tensor(batch["x_video"]):
            B = int(batch["x_video"].shape[0])
        else:
            raise KeyError("x_video in batch).")

        # 3) if user specified a key, try it; else auto infer
        uids: Optional[List[str]] = None
        if self.batch_uid_key != "auto" and self.batch_uid_key in batch:
            uids = self._try_make_uid_list(batch.get(self.batch_uid_key, None), B)

        if uids is None:
            uids = self._infer_uids_from_batch(batch, B)

        xm = self.meta_table.lookup(uids, device=device)  # [B,Dm]
        xm = torch.nan_to_num(xm, nan=0.0, posinf=0.0, neginf=0.0)
        return xm

    def forward_with_extras(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        xa = batch["x_audio"]
        ma = batch.get("x_audio_mask", None)
        xv = batch["x_video"]
        mv = batch.get("x_video_mask", None)

        device = xa.device

        ha = self._encode(xa, ma, self.proj_a, self.enc_a)
        hv = self._encode(xv, mv, self.proj_v, self.enc_v)

        pa, attn_a = self.pool_a(ha, ma)  # [B,2H]
        pv, attn_v = self.pool_v(hv, mv)  # [B,2H]

        pooled = torch.cat([pa, pv], dim=-1)  # [B,4H]
        extras: Dict[str, torch.Tensor] = {"attn_audio": attn_a, "attn_video": attn_v}

        if self.use_meta:
            xm = self._get_x_meta(batch, device=device)  # [B,Dm]
            pm = self.proj_m(xm)  # [B,meta_out]
            pooled = torch.cat([pooled, pm], dim=-1)
            extras["x_meta"] = xm
            extras["meta_proj"] = pm

        logits = self.cls(pooled)
        return logits, pooled, extras

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        logits, _, _ = self.forward_with_extras(batch)
        return logits
