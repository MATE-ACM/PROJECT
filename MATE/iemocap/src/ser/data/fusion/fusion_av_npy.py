from __future__ import annotations

"""src.ser.data.fusion.fusion_av_npy

Audio-Visual fusion dataset for *pre-extracted frame-level* features.

Per utterance files:
  Audio: <audio_feat_root>/<utt_id>.<audio_ext>  (default .npy)  -> [Ta, Da]
  Video: <video_feat_root>/<utt_id>.<video_ext>  (default .npz)  -> [Tv, Dv]
        - .npy: array [Tv, Dv]
        - .npz: expects key `video_feat_key` (default "x") storing [Tv, Dv]

Manifest item (dict) must contain:
  - utt_id (str)
  - label  (str)

Collate outputs:
  - utt_id: list[str]
  - x_audio:      [B, Ta, Da]
  - x_audio_mask: [B, Ta]  (1=valid, 0=pad)
  - x_video:      [B, Tv, Dv]
  - x_video_mask: [B, Tv]  (1=valid, 0=pad)
  - y:            [B]

Windows note:
  - collate_* is a top-level function (picklable), so num_workers>0 works.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _norm_ext(ext: str) -> str:
    ext = str(ext)
    return ext if ext.startswith(".") else ("." + ext)


def _resolve_feat_path(root: str, utt_id: str, ext: str) -> str:
    ext = _norm_ext(ext)
    if utt_id.endswith(ext):
        return os.path.join(str(root), utt_id)
    return os.path.join(str(root), f"{utt_id}{ext}")


def _load_2d(path: str, ext: str, npz_key: str = "x") -> np.ndarray:
    ext = _norm_ext(ext).lower()
    if ext == ".npy":
        x = np.load(path)
    elif ext == ".npz":
        z = np.load(path, allow_pickle=True)
        if npz_key in z:
            x = z[npz_key]
        else:
            keys = list(z.files)
            if len(keys) == 1:
                x = z[keys[0]]
            else:
                raise KeyError(f"NPZ missing key='{npz_key}'. Available keys={keys}. File={path}")
    else:
        raise ValueError(f"Unsupported ext={ext}. Use .npy or .npz")

    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected [T,D], got {x.shape} from {path}")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x


def _trim_or_pad_2d(x: torch.Tensor, target_len: int, pad_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim/pad a [T,D] tensor to [target_len,D], returning (x_out, mask)."""
    assert x.ndim == 2, f"expected [T,D], got {tuple(x.shape)}"
    T, D = x.shape

    if target_len <= 0:
        return x[:0], x.new_zeros((0,), dtype=torch.long)

    if T == target_len:
        return x, x.new_ones((target_len,), dtype=torch.long)

    if T > target_len:
        if pad_mode == "center":
            s = (T - target_len) // 2
        elif pad_mode == "left":
            s = T - target_len
        else:
            s = 0
        x2 = x[s:s + target_len]
        return x2, x2.new_ones((target_len,), dtype=torch.long)

    # pad
    out = x.new_zeros((target_len, D))
    mask = x.new_zeros((target_len,), dtype=torch.long)
    if pad_mode == "center":
        s = (target_len - T) // 2
    elif pad_mode == "left":
        s = target_len - T
    else:
        s = 0
    out[s:s + T] = x
    mask[s:s + T] = 1
    return out, mask


class FusionAVNpyDataset(Dataset):
    def __init__(
        self,
        items: List[Dict[str, Any]],
        label2id: Dict[str, int],
        audio_feat_root: str,
        video_feat_root: str,
        audio_ext: str = ".npy",
        video_ext: str = ".npz",
        video_feat_key: str = "x",
        audio_input_dim: Optional[int] = None,
        video_input_dim: Optional[int] = None,
        allow_missing_audio: bool = False,
        allow_missing_video: bool = False,
    ):
        self.items = items
        self.label2id = label2id
        self.audio_feat_root = str(audio_feat_root)
        self.video_feat_root = str(video_feat_root)
        self.audio_ext = str(audio_ext)
        self.video_ext = str(video_ext)
        self.video_feat_key = str(video_feat_key)
        self.audio_input_dim = int(audio_input_dim) if audio_input_dim is not None else None
        self.video_input_dim = int(video_input_dim) if video_input_dim is not None else None
        self.allow_missing_audio = bool(allow_missing_audio)
        self.allow_missing_video = bool(allow_missing_video)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        utt_id = str(it["utt_id"])
        label = str(it["label"])
        if label not in self.label2id:
            raise KeyError(f"Unknown label={label}. label2id keys={list(self.label2id.keys())}")
        y = int(self.label2id[label])

        # --- audio ---
        # Preferred: <audio_feat_root>/<utt_id>.<ext>
        # Fallbacks: <audio_feat_root>/<speaker>/<utt_id>.<ext> or <speaker>/<cut_utt_id>.<ext>
        cand_ids = [utt_id]
        spk = it.get("speaker", None)
        if spk is not None:
            spk = str(spk)
            cand_ids.append(f"{spk}/{utt_id}")
            if utt_id.startswith(spk + "_"):
                cut_utt_id = utt_id[len(spk) + 1 :]
                cand_ids.append(f"{spk}/{cut_utt_id}")

        a_path = None
        tried_a = []
        for cid in cand_ids:
            p = _resolve_feat_path(self.audio_feat_root, cid, self.audio_ext)
            tried_a.append(p)
            if os.path.exists(p):
                a_path = p
                break

        if a_path is None:
            if self.allow_missing_audio:
                a = np.zeros((1, self.audio_input_dim or 1), dtype=np.float32)
                a_mask = np.zeros((1,), dtype=np.int64)
            else:
                preview = "\n".join(tried_a[:5])
                raise FileNotFoundError(
                    f"n{preview}"
                )
        else:
            a = _load_2d(a_path, ext=self.audio_ext)
            if self.audio_input_dim is not None and int(a.shape[1]) != self.audio_input_dim:
                raise ValueError(f"audio_input_dim mismatch: cfg={self.audio_input_dim}, file D={a.shape[1]} @ {a_path}")
            a_mask = np.ones((a.shape[0],), dtype=np.int64)

        # --- video ---
        v_path = _resolve_feat_path(self.video_feat_root, utt_id, self.video_ext)
        if not os.path.exists(v_path):
            if self.allow_missing_video:
                v = np.zeros((1, self.video_input_dim or 1), dtype=np.float32)
                v_mask = np.zeros((1,), dtype=np.int64)
            else:
                raise FileNotFoundError(f"Missing video feature file: {v_path}")
        else:
            v = _load_2d(v_path, ext=self.video_ext, npz_key=self.video_feat_key)
            if self.video_input_dim is not None and int(v.shape[1]) != self.video_input_dim:
                raise ValueError(f"video_input_dim mismatch: cfg={self.video_input_dim}, file D={v.shape[1]} @ {v_path}")
            v_mask = np.ones((v.shape[0],), dtype=np.int64)

        return {
            "utt_id": utt_id,
            "x_audio": torch.from_numpy(a),
            "x_audio_mask": torch.from_numpy(a_mask),
            "x_video": torch.from_numpy(v),
            "x_video_mask": torch.from_numpy(v_mask),
            "y": y,
        }


def collate_fusion_av_npy(
    items: List[Dict[str, Any]],
    pad_mode: str = "right",
    max_frames_audio: Optional[int] = None,
    max_frames_video: Optional[int] = None,
) -> Dict[str, Any]:
    utt_id = [str(it["utt_id"]) for it in items]
    y = torch.tensor([int(it["y"]) for it in items], dtype=torch.long)

    # audio
    a_list = [it["x_audio"] for it in items]
    a_len = [int(x.shape[0]) for x in a_list]
    Ta = max(a_len) if a_len else 0
    if max_frames_audio is not None:
        Ta = min(Ta, int(max_frames_audio))
    Da = int(a_list[0].shape[1]) if a_list else 0
    a_out = a_list[0].new_zeros((len(a_list), Ta, Da)) if a_list else torch.zeros((0, 0, 0))
    am_out = torch.zeros((len(a_list), Ta), dtype=torch.long)

    # video
    v_list = [it["x_video"] for it in items]
    v_len = [int(x.shape[0]) for x in v_list]
    Tv = max(v_len) if v_len else 0
    if max_frames_video is not None:
        Tv = min(Tv, int(max_frames_video))
    Dv = int(v_list[0].shape[1]) if v_list else 0
    v_out = v_list[0].new_zeros((len(v_list), Tv, Dv)) if v_list else torch.zeros((0, 0, 0))
    vm_out = torch.zeros((len(v_list), Tv), dtype=torch.long)

    for i in range(len(items)):
        # audio
        xa = a_list[i][:Ta]
        xa2, ma = _trim_or_pad_2d(xa, Ta, pad_mode=pad_mode)
        a_out[i] = xa2
        am_out[i] = ma

        # video
        xv = v_list[i][:Tv]
        xv2, mv = _trim_or_pad_2d(xv, Tv, pad_mode=pad_mode)
        v_out[i] = xv2
        vm_out[i] = mv

    return {
        "utt_id": utt_id,
        "x_audio": a_out,
        "x_audio_mask": am_out,
        "x_video": v_out,
        "x_video_mask": vm_out,
        "y": y,
    }
