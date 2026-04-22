from __future__ import annotations

"""src.ser.data.txt.txt_npy

Text feature dataset + collate for per-utterance saved numpy features.

Conventions (same spirit as audio/video in this repo):
- Each utterance has one file: <feat_root>/<utt_id>.npy (or .npz)
- Feature array can be:
    - [D]        (utterance-level embedding, e.g. CLS / mean pool)
    - [T, D]     (token-level / frame-level sequence)
- We keep variable length inside Dataset and do padding + mask in collate.

Outputs (collate):
  - utt_id: list[str]
  - x_txt: [B, T, D]
  - x_txt_mask: [B, T]  (1=valid, 0=pad)
  - y: [B]

NOTE: keys starting with "x_" will be automatically moved to GPU in scripts/train.py.
"""

from typing import Any, Dict, List, Optional, Tuple
import os

import numpy as np
import torch
from torch.utils.data import Dataset

def _get_item_label(it):
    lb = it.get("label", None)
    if lb is None:
        lb = it.get("emotion", None)
    if lb is None:
        raise KeyError(f"Sample missing both 'label' and 'emotion'. keys={list(it.keys())}")
    return lb

def _resolve_feat_path(feat_root: str, utt_id: str, ext: str = ".npy") -> str:
    """Join feat_root + utt_id (+ ext if needed)."""
    ext = ext if ext.startswith(".") else ("." + ext)
    if utt_id.endswith(ext):
        return os.path.join(str(feat_root), utt_id)
    return os.path.join(str(feat_root), f"{utt_id}{ext}")

def _load_feat(path: str, ext: str, feat_key: str = "x", mmap: bool = False) -> np.ndarray:
    ext = ext.lower()
    if ext == ".npy":
        return np.load(path, mmap_mode="r" if mmap else None)
    if ext == ".npz":
        z = np.load(path, allow_pickle=True)
        if feat_key in z:
            return z[feat_key]
        keys = list(z.files)
        if len(keys) == 1:
            return z[keys[0]]
        raise KeyError(f"NPZ missing key='{feat_key}'. Available keys={keys}")
    raise ValueError(f"Unsupported ext={ext}. Use .npy or .npz")

def _trim_or_pad_2d(
    x: torch.Tensor,
    target_len: int,
    pad_mode: str = "right",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim/pad [T,D] -> [target_len,D] + mask [target_len]."""
    assert x.ndim == 2, f"expected [T,D], got {tuple(x.shape)}"
    T, D = x.shape

    if target_len <= 0:
        return x[:0], x.new_zeros((0,), dtype=torch.long)

    if T == target_len:
        return x, x.new_ones((target_len,), dtype=torch.long)

    # trim
    if T > target_len:
        if pad_mode == "center":
            start = (T - target_len) // 2
        elif pad_mode == "left":
            start = T - target_len
        else:  # right/default -> keep prefix
            start = 0
        x_out = x[start : start + target_len]
        mask = x_out.new_ones((target_len,), dtype=torch.long)
        return x_out, mask

    # pad
    pad = target_len - T
    if pad_mode == "left":
        x_out = torch.cat([x.new_zeros((pad, D)), x], dim=0)
        mask = torch.cat(
            [x.new_zeros((pad,), dtype=torch.long), x.new_ones((T,), dtype=torch.long)],
            dim=0,
        )
        return x_out, mask

    if pad_mode == "center":
        left = pad // 2
        right = pad - left
        x_out = torch.cat([x.new_zeros((left, D)), x, x.new_zeros((right, D))], dim=0)
        mask = torch.cat(
            [
                x.new_zeros((left,), dtype=torch.long),
                x.new_ones((T,), dtype=torch.long),
                x.new_zeros((right,), dtype=torch.long),
            ],
            dim=0,
        )
        return x_out, mask

    # right/default
    x_out = torch.cat([x, x.new_zeros((pad, D))], dim=0)
    mask = torch.cat(
        [x.new_ones((T,), dtype=torch.long), x.new_zeros((pad,), dtype=torch.long)],
        dim=0,
    )
    return x_out, mask

class TxtNpyDataset(Dataset):
    """Per-utterance text features saved as .npy/.npz.

    Manifest item should contain:
      - utt_id: str
      - label:  str (or digit str)
      - speaker: (optional) used for fallback subfolder search

    Output item:
      - x_txt: torch.Tensor [T,D] (unpadded)
      - y: int
    """

    def __init__(
        self,
        items: List[Dict[str, Any]],
        label2id: Dict[Any, int],
        feat_root: str,
        ext: str = ".npy",
        feat_key: str = "x",
        pad_mode: str = "right",
        max_frames: Optional[int] = None,
        mmap: bool = False,
        input_dim: Optional[int] = None,
    ):
        self.items = items
        self.label2id = label2id
        self.feat_root = str(feat_root)
        self.ext = str(ext)
        self.feat_key = str(feat_key)
        self.pad_mode = str(pad_mode)
        self.max_frames = int(max_frames) if max_frames is not None else None
        self.mmap = bool(mmap)
        self.input_dim = int(input_dim) if input_dim is not None else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        utt_id = str(it["utt_id"])
        label = it.get("label", it.get("emotion"))

        if label not in self.label2id and str(label) not in self.label2id:
            raise KeyError(f"Unknown label={label}. label2id keys sample={list(self.label2id.keys())[:20]}")
        y = int(self.label2id.get(label, self.label2id[str(label)]))

        # Preferred: <feat_root>/<utt_id>.<ext>
        # Fallbacks: <feat_root>/<speaker>/<utt_id>.<ext> etc.
        cand_ids = [utt_id]
        spk = it.get("speaker", None)
        if spk is not None:
            spk = str(spk)
            cand_ids.append(f"{spk}/{utt_id}")
            if utt_id.startswith(spk + "_"):
                cut_utt_id = utt_id[len(spk) + 1 :]
                cand_ids.append(f"{spk}/{cut_utt_id}")

        path = None
        tried: List[str] = []
        for cid in cand_ids:
            p = _resolve_feat_path(self.feat_root, cid, ext=self.ext)
            tried.append(p)
            if os.path.exists(p):
                path = p
                break

        if path is None:
            preview = "\n".join(tried[:5])
            raise FileNotFoundError(
                f"n{preview}"
            )

        arr = _load_feat(path, ext=self.ext, feat_key=self.feat_key, mmap=self.mmap)
        arr = np.asarray(arr)

        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"Expected feature shape [D] or [T,D], got {arr.shape} @ {path}")

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if self.input_dim is not None and int(arr.shape[1]) != self.input_dim:
            raise ValueError(
                f"[{utt_id}] feature dim mismatch: got {arr.shape[1]}, expect {self.input_dim}. "
                f"Check expert.input_dim and feat_root. ({path})"
            )

        x = torch.from_numpy(arr)  # [T,D]
        if self.max_frames is not None and x.shape[0] > self.max_frames:
            x = x[: self.max_frames]

        return {
            "utt_id": utt_id,
            "x_txt": x,
            "y": y,
        }

def collate_txt_npy(
    batch: List[Dict[str, Any]],
    pad_mode: str = "right",
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Pad variable-length text feature sequences in a batch."""
    utt_id = [b["utt_id"] for b in batch]
    y = torch.tensor([int(b["y"]) for b in batch], dtype=torch.long)

    feats = [b["x_txt"] for b in batch]
    lengths = [int(f.shape[0]) for f in feats]
    T_max = max(lengths) if lengths else 0
    if max_frames is not None:
        T_max = min(T_max, int(max_frames))

    D = int(feats[0].shape[1]) if feats else 0
    x_out = feats[0].new_zeros((len(feats), T_max, D)) if feats else torch.zeros((0, 0, 0))
    m_out = torch.zeros((len(feats), T_max), dtype=torch.long)

    for i, f in enumerate(feats):
        f = f[:T_max]
        f_pad, mask = _trim_or_pad_2d(f, T_max, pad_mode=pad_mode)
        x_out[i] = f_pad
        m_out[i] = mask

    return {"utt_id": utt_id, "x_txt": x_out, "x_txt_mask": m_out, "y": y}
