from __future__ import annotations

"""Dataset and collate helpers for frame-level audio npy files."""

from typing import Dict, Any, List, Optional, Tuple
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
    """Build a feature path from utt_id. Supports utt_id values that already include an extension."""

    if utt_id.endswith(ext):
        return os.path.join(feat_root, utt_id)
    return os.path.join(feat_root, f"{utt_id}{ext}")

def _trim_or_pad_2d(
    x: torch.Tensor,
    target_len: int,
    pad_mode: str = "right",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim or pad a [T, D] array and return (x_out, mask).

    Args:
        x: [T, D]
        target_len: target sequence length
        pad_mode: right|left|center

    Returns:
        x_out: [target_len, D]
        mask : [target_len] long, 1=valid, 0=pad
    """
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
        else:  # right/default: keep prefix
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

class AudioNpyDataset(Dataset):
    """Read per-utterance frame-level .npy features.

    Expected manifest fields:
      - utt_id: used to resolve <feat_root>/<utt_id>.npy
      - label: string label
      - speaker_id: optional speaker id for speaker-disjoint splits
    """

    def __init__(
        self,
        items: List[Dict[str, Any]],
        label2id: Dict[str, int],
        feat_root: str,
        ext: str = ".npy",
        pad_mode: str = "right",
        max_frames: Optional[int] = None,
        mmap: bool = False,
        input_dim: Optional[int] = None,
    ):
        self.items = items
        self.label2id = label2id
        self.feat_root = feat_root
        self.ext = ext
        self.pad_mode = pad_mode
        self.max_frames = int(max_frames) if max_frames is not None else None
        self.mmap = bool(mmap)
        self.input_dim = int(input_dim) if input_dim is not None else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        utt_id = str(it["utt_id"])
        label = str(it.get("label", it.get("emotion")))

        if label not in self.label2id:
            raise KeyError(f"Unknown label={label}. label2id keys={list(self.label2id.keys())}")
        y = int(self.label2id[label])

        # IEMOCAP/CREMA-D feature conventions in this repo:
        #   1) preferred: <feat_root>/<utt_id>.<ext>
        # Some old scripts may save under speaker subfolders, so we provide a
        # small set of fallbacks to make migration smoother.
        cand_ids = [utt_id]
        spk = it.get("speaker", None)
        if spk is not None:
            spk = str(spk)
            # <feat_root>/<speaker>/<utt_id>.<ext>
            cand_ids.append(f"{spk}/{utt_id}")
            # <feat_root>/<speaker>/<cut_utt_id>.<ext>  where cut removes "<speaker>_" prefix
            if utt_id.startswith(spk + "_"):
                cut_utt_id = utt_id[len(spk) + 1 :]
                cand_ids.append(f"{spk}/{cut_utt_id}")

        path = None
        tried = []
        for cid in cand_ids:
            p = _resolve_feat_path(self.feat_root, cid, ext=self.ext)
            tried.append(p)
            if os.path.exists(p):
                path = p
                break

        if path is None:
            preview = "\n".join(tried[:5])
            raise FileNotFoundError(
                f"Feature file not found. Tried:\n{preview}"
            )

        arr = np.load(path, mmap_mode="r" if self.mmap else None)
        if arr.ndim == 1:
            arr = arr[None, :]
        arr = np.asarray(arr, dtype=np.float32)

        if self.input_dim is not None and arr.shape[1] != self.input_dim:
            raise ValueError(
                f"[{utt_id}] feature dim mismatch: got {arr.shape[1]}, expect {self.input_dim}. "
                f"Check expert.input_dim and feat_root."
            )

        x = torch.from_numpy(arr)  # [T,D]
        if self.max_frames is not None and x.shape[0] > self.max_frames:
            x = x[: self.max_frames]

        return {
            "utt_id": utt_id,
            "x_audio": x,  # [T,D] unpadded
            "y": y,
            "speaker_id": it.get("speaker_id", None),
        }

def collate_audio_npy(
    batch: List[Dict[str, Any]],
    pad_mode: str = "right",
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Pad variable-length sequences to the batch maximum length.

    Returns:
      x_audio: [B,T,D]
      x_audio_mask: [B,T] (1 valid, 0 pad)

    NOTE: keys starting with "x_" are automatically moved to GPU in scripts/train.py.
    """
    utt_id = [b["utt_id"] for b in batch]
    y = torch.tensor([int(b["y"]) for b in batch], dtype=torch.long)

    feats = [b["x_audio"] for b in batch]
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

    return {"utt_id": utt_id, "x_audio": x_out, "x_audio_mask": m_out, "y": y}
