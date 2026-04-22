from __future__ import annotations

"""src.ser.data.fusion.fusion_avt_npy

Audio-Visual-Text fusion dataset for *pre-extracted frame/token-level* features.

Per utterance files:
  Audio: <audio_feat_root>/<utt_id>.<audio_ext>   (.npy / .npz) -> [Ta, Da]
  Video: <video_feat_root>/<utt_id>.<video_ext>   (.npy / .npz) -> [Tv, Dv]
  Text : <text_feat_root>/<utt_id>.<text_ext>    (.npy / .npz) -> [Tt, Dt] or [Dt] (will be treated as [1,Dt])

Manifest item (dict) must contain:
  - utt_id (str)
  - label  (str)
Optional:
  - speaker (str): used as fallback path <root>/<speaker>/<utt_id>.<ext>

Returns (per item):
  - x_audio [Ta,Da], x_audio_mask [Ta]
  - x_video [Tv,Dv], x_video_mask [Tv]
  - x_text  [Tt,Dt], x_text_mask  [Tt]
  - y (int)

Padding/truncation is performed in collate_fusion_avt_npy.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

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

def _norm_ext(ext: str) -> str:
    ext = str(ext)
    if not ext.startswith("."):
        ext = "." + ext
    return ext

def _resolve_feat_path(root: str, utt_id: str, ext: str) -> str:
    return os.path.join(str(root), f"{utt_id}{_norm_ext(ext)}")

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

    # allow utterance-level embeddings [D]
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected [T,D] or [D], got {x.shape} from {path}")
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

class FusionAVTNpyDataset(Dataset):
    def __init__(
        self,
        items: List[Dict[str, Any]],
        label2id: Dict[str, int],
        audio_feat_root: str,
        video_feat_root: str,
        text_feat_root: str,
        audio_ext: str = ".npy",
        video_ext: str = ".npz",
        text_ext: str = ".npy",
        video_feat_key: str = "x",
        text_feat_key: str = "x",
        audio_input_dim: Optional[int] = None,
        video_input_dim: Optional[int] = None,
        text_input_dim: Optional[int] = None,
        allow_missing_audio: bool = False,
        allow_missing_video: bool = False,
        allow_missing_text: bool = False,
    ):
        self.items = items
        self.label2id = label2id
        self.audio_feat_root = str(audio_feat_root)
        self.video_feat_root = str(video_feat_root)
        self.text_feat_root = str(text_feat_root)

        self.audio_ext = str(audio_ext)
        self.video_ext = str(video_ext)
        self.text_ext = str(text_ext)

        self.video_feat_key = str(video_feat_key)
        self.text_feat_key = str(text_feat_key)

        self.audio_input_dim = int(audio_input_dim) if audio_input_dim is not None else None
        self.video_input_dim = int(video_input_dim) if video_input_dim is not None else None
        self.text_input_dim = int(text_input_dim) if text_input_dim is not None else None

        self.allow_missing_audio = bool(allow_missing_audio)
        self.allow_missing_video = bool(allow_missing_video)
        self.allow_missing_text = bool(allow_missing_text)

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_with_fallbacks(self, root: str, utt_id: str, ext: str, speaker: Optional[str]) -> Tuple[Optional[str], List[str]]:
        cand_ids = [utt_id]
        if speaker:
            spk = str(speaker)
            cand_ids.append(f"{spk}/{utt_id}")
            if utt_id.startswith(spk + "_"):
                cut_utt = utt_id[len(spk) + 1 :]
                cand_ids.append(f"{spk}/{cut_utt}")
        tried = []
        for cid in cand_ids:
            p = _resolve_feat_path(root, cid, ext)
            tried.append(p)
            if os.path.exists(p):
                return p, tried
        return None, tried

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        utt_id = str(it["utt_id"])
        label = str(it.get("label", it.get("emotion")))
        if label not in self.label2id:
            raise KeyError(f"Unknown label={label}. label2id keys={list(self.label2id.keys())}")
        y = int(self.label2id[label])

        speaker = it.get("speaker", None)

        # ---- audio ----
        a_path, tried_a = self._resolve_with_fallbacks(self.audio_feat_root, utt_id, self.audio_ext, speaker)
        if a_path is None:
            if not self.allow_missing_audio:
                raise FileNotFoundError(f"Audio feature not found for utt_id={utt_id}. Tried={tried_a[:3]}... (total {len(tried_a)})")
            Da = self.audio_input_dim or 1
            a = np.zeros((1, Da), dtype=np.float32)
        else:
            a = _load_2d(a_path, self.audio_ext, npz_key="x")
        if self.audio_input_dim is not None and a.shape[1] != self.audio_input_dim:
            raise ValueError(f"Audio dim mismatch: got {a.shape[1]}, expected {self.audio_input_dim}. File={a_path}")

        # ---- video ----
        v_path, tried_v = self._resolve_with_fallbacks(self.video_feat_root, utt_id, self.video_ext, speaker)
        if v_path is None:
            if not self.allow_missing_video:
                raise FileNotFoundError(f"Video feature not found for utt_id={utt_id}. Tried={tried_v[:3]}... (total {len(tried_v)})")
            Dv = self.video_input_dim or 1
            v = np.zeros((1, Dv), dtype=np.float32)
        else:
            v = _load_2d(v_path, self.video_ext, npz_key=self.video_feat_key)
        if self.video_input_dim is not None and v.shape[1] != self.video_input_dim:
            raise ValueError(f"Video dim mismatch: got {v.shape[1]}, expected {self.video_input_dim}. File={v_path}")

        # ---- text ----
        t_path, tried_t = self._resolve_with_fallbacks(self.text_feat_root, utt_id, self.text_ext, speaker)
        if t_path is None:
            if not self.allow_missing_text:
                raise FileNotFoundError(f"Text feature not found for utt_id={utt_id}. Tried={tried_t[:3]}... (total {len(tried_t)})")
            Dt = self.text_input_dim or 1
            t = np.zeros((1, Dt), dtype=np.float32)
        else:
            t = _load_2d(t_path, self.text_ext, npz_key=self.text_feat_key)
        if self.text_input_dim is not None and t.shape[1] != self.text_input_dim:
            raise ValueError(f"Text dim mismatch: got {t.shape[1]}, expected {self.text_input_dim}. File={t_path}")

        a_t = torch.from_numpy(a)
        v_t = torch.from_numpy(v)
        t_t = torch.from_numpy(t)

        return {
            "utt_id": utt_id,
            "x_audio": a_t,
            "x_audio_mask": torch.ones((a_t.shape[0],), dtype=torch.long),
            "x_video": v_t,
            "x_video_mask": torch.ones((v_t.shape[0],), dtype=torch.long),
            "x_text": t_t,
            "x_text_mask": torch.ones((t_t.shape[0],), dtype=torch.long),
            "y": y,
        }

def collate_fusion_avt_npy(
    items: List[Dict[str, Any]],
    pad_mode: str = "right",
    max_frames_audio: Optional[int] = None,
    max_frames_video: Optional[int] = None,
    max_frames_text: Optional[int] = None,
) -> Dict[str, Any]:
    utt_ids = [it["utt_id"] for it in items]
    ys = torch.tensor([it["y"] for it in items], dtype=torch.long)

    xa = [it["x_audio"] for it in items]
    xv = [it["x_video"] for it in items]
    xt = [it["x_text"] for it in items]

    Ta = max(x.shape[0] for x in xa) if max_frames_audio is None else int(max_frames_audio)
    Tv = max(x.shape[0] for x in xv) if max_frames_video is None else int(max_frames_video)
    Tt = max(x.shape[0] for x in xt) if max_frames_text is None else int(max_frames_text)

    xa2, ma2 = zip(*[_trim_or_pad_2d(x, Ta, pad_mode) for x in xa])
    xv2, mv2 = zip(*[_trim_or_pad_2d(x, Tv, pad_mode) for x in xv])
    xt2, mt2 = zip(*[_trim_or_pad_2d(x, Tt, pad_mode) for x in xt])

    return {
        "utt_id": utt_ids,
        "x_audio": torch.stack(list(xa2), dim=0),
        "x_audio_mask": torch.stack(list(ma2), dim=0),
        "x_video": torch.stack(list(xv2), dim=0),
        "x_video_mask": torch.stack(list(mv2), dim=0),
        "x_text": torch.stack(list(xt2), dim=0),
        "x_text_mask": torch.stack(list(mt2), dim=0),
        "y": ys,
    }
