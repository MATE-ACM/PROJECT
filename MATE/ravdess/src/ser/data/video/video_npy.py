from __future__ import annotations

"""
Video frame-level features dataset (.npy / .npz) + collate (pad + mask).

Per utterance file:
  <feat_root>/<utt_id>.{npy|npz}
  - npy: array [T, D]
  - npz: expects key `feat_key` (default "x") storing [T, D]

Outputs (collate):
  - utt_id: list[str]
  - x_video: [B, T, D]
  - x_video_mask: [B, T]  (1=valid, 0=pad)
  - y: [B]
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _resolve_feat_path(feat_root: str, utt_id: str, ext: str) -> str:
    ext = ext if ext.startswith(".") else ("." + ext)
    return os.path.join(str(feat_root), f"{utt_id}{ext}")


def _load_feat(path: str, ext: str, feat_key: str) -> np.ndarray:
    ext = ext.lower()
    if ext == ".npy":
        x = np.load(path)
        return x
    if ext == ".npz":
        z = np.load(path, allow_pickle=True)
        if feat_key in z:
            return z[feat_key]
        # fallback: if only one array exists
        keys = list(z.files)
        if len(keys) == 1:
            return z[keys[0]]
        raise KeyError(f"NPZ missing key='{feat_key}'. Available keys={keys}")
    raise ValueError(f"Unsupported ext={ext}. Use .npy or .npz")


def _trim_or_pad_2d(x: torch.Tensor, target_len: int, pad_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 2, f"expected [T,D], got {tuple(x.shape)}"
    T, D = x.shape

    if target_len <= 0:
        return x[:0], x.new_zeros((0,), dtype=torch.long)

    if T == target_len:
        return x, x.new_ones((target_len,), dtype=torch.long)

    if T > target_len:
        if pad_mode == "center":
            s = (T - target_len) // 2
            x2 = x[s:s + target_len]
        else:
            x2 = x[:target_len]
        return x2, x2.new_ones((target_len,), dtype=torch.long)

    # pad
    out = x.new_zeros((target_len, D))
    mask = x.new_zeros((target_len,), dtype=torch.long)
    if pad_mode == "center":
        s = (target_len - T) // 2
    else:
        s = 0
    out[s:s + T] = x
    mask[s:s + T] = 1
    return out, mask


class VideoNpyDataset(Dataset):
    def __init__(
        self,
        items: List[Dict[str, Any]],
        label2id: Dict[str, int],
        feat_root: str,
        ext: str = ".npz",
        feat_key: str = "x",
        pad_mode: str = "right",
        max_frames: Optional[int] = None,
        input_dim: Optional[int] = None,
    ):
        self.items = items
        self.label2id = label2id
        self.feat_root = str(feat_root)
        self.ext = str(ext)
        self.feat_key = str(feat_key)
        self.pad_mode = str(pad_mode)
        self.max_frames = int(max_frames) if max_frames is not None else None
        self.input_dim = int(input_dim) if input_dim is not None else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        utt_id = str(it["utt_id"])
        label = str(it["label"])

        if label not in self.label2id:
            raise KeyError(f"Unknown label={label}. label2id keys={list(self.label2id.keys())}")
        y = int(self.label2id[label])
        #shared_code = utt_id[-17:]
        #shared_code='01-'+shared_code  这个代码是专门用来处理openface特征的逻辑 因为openface的命名逻辑跟hsemotion的不一样
        #path = _resolve_feat_path(self.feat_root, shared_code, ext=self.ext)
        path = _resolve_feat_path(self.feat_root, utt_id, ext=self.ext)
        print(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing video feature file: {path}")

        x = _load_feat(path, ext=self.ext, feat_key=self.feat_key)
        #x = x[:, :35] 只有在openface时才使用
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2:
            raise ValueError(f"Expected feature shape [T,D], got {x.shape} from {path}")

        if self.input_dim is not None and int(x.shape[1]) != self.input_dim:
            raise ValueError(f"input_dim mismatch: cfg={self.input_dim}, file D={x.shape[1]} @ {path}")

        # do not pad here; keep variable length, pad in collate
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        x_t = torch.from_numpy(x)  # [T,D]
        return {"utt_id": utt_id, "x": x_t, "y": y}


def collate_video_npy(items: List[Dict[str, Any]], pad_mode: str = "right", max_frames: Optional[int] = None) -> Dict[str, Any]:
    utt_id = [str(it["utt_id"]) for it in items]
    y = torch.tensor([int(it["y"]) for it in items], dtype=torch.long)

    feats = [it["x"] for it in items]  # [T,D] tensors
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

    return {"utt_id": utt_id, "x_video": x_out, "x_video_mask": m_out, "y": y}
