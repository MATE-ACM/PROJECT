from __future__ import annotations

"""src.ser.data.audio_npy

【文件作用】读取 .npy 帧级特征的数据集与 collate（支持 padding + mask）。

设计要点：
- 每条样本一个文件：<feat_root>/<utt_id>.npy
- 支持变长序列：collate 时 padding，并输出 mask（1=valid,0=pad）
- （可选）input_dim 维度校验：防止特征目录/配置写错导致“悄悄训练错维度”
"""


from typing import Dict, Any, List, Optional, Tuple
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def _resolve_feat_path(feat_root: str, utt_id: str, ext: str = ".npy") -> str:
    """根据 utt_id 拼出特征路径。支持 utt_id 已包含 ext。"""
    if utt_id.endswith(ext):
        return os.path.join(feat_root, utt_id)
    return os.path.join(feat_root, f"{utt_id}{ext}")


def _trim_or_pad_2d(
    x: torch.Tensor,
    target_len: int,
    pad_mode: str = "right",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 [T,D] 进行 trim/pad，返回 (x_out, mask)。

    Args:
        x: [T, D]
        target_len: 目标长度
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
    """读取 per-utterance 的帧级 .npy 特征。

    Manifest item (dict) 需要：
      - utt_id: 用来定位 <feat_root>/<utt_id>.npy
      - label: 字符串情绪标签
      - speaker_id: (可选) 供 speaker-disjoint split
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
        label = str(it["label"])

        if label not in self.label2id:
            raise KeyError(f"Unknown label={label}. label2id keys={list(self.label2id.keys())}")
        y = int(self.label2id[label])

        path = _resolve_feat_path(self.feat_root, utt_id, ext=self.ext)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing feature file: {path}")

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
    """把变长序列 padding 到 batch 内最大长度（可选 cap）。

    Returns:
      x_audio: [B,T,D]
      x_audio_mask: [B,T] (1 valid, 0 pad)

    NOTE: key 以 "x_" 开头，scripts/train.py 会自动搬到 GPU。
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
