from __future__ import annotations

from functools import partial
from typing import Any, Dict, List

from torch.utils.data import DataLoader

from .fusion_av_npy import FusionAVNpyDataset, collate_fusion_av_npy


def make_fusion_loaders(
    items_train: List[Dict[str, Any]],
    items_val: List[Dict[str, Any]],
    items_test: List[Dict[str, Any]],
    label2id: Dict[str, int],
    cfg: Dict[str, Any],
    expert_type: str,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Build DataLoaders for fusion_* experts.

    Currently supports:
      - fusion_av_* : audio + video frame-level features
    """
    e = cfg.get("expert", {}) or {}

    audio_feat_root = e.get("audio_feat_root")
    video_feat_root = e.get("video_feat_root")
    if audio_feat_root is None or video_feat_root is None:
        raise KeyError(
            f"video_feat_root in expert.*."
            "Example: expert.audio_feat_root, expert.video_feat_root"
        )

    audio_ext = str(e.get("audio_ext", e.get("ext_audio", ".npy")))
    video_ext = str(e.get("video_ext", e.get("ext_video", ".npz")))
    video_feat_key = str(e.get("video_feat_key", e.get("feat_key_video", "x")))

    audio_input_dim = e.get("audio_input_dim")
    video_input_dim = e.get("video_input_dim")
    audio_input_dim = int(audio_input_dim) if audio_input_dim is not None else None
    video_input_dim = int(video_input_dim) if video_input_dim is not None else None

    pad_mode = str(e.get("pad_mode", "right"))
    max_frames_audio = e.get("max_frames_audio", e.get("max_frames", 600))
    max_frames_video = e.get("max_frames_video", 300)
    max_frames_audio = int(max_frames_audio) if max_frames_audio is not None else None
    max_frames_video = int(max_frames_video) if max_frames_video is not None else None

    allow_missing_audio = bool(e.get("allow_missing_audio", False))
    allow_missing_video = bool(e.get("allow_missing_video", False))

    ds_tr = FusionAVNpyDataset(
        items_train,
        label2id=label2id,
        audio_feat_root=str(audio_feat_root),
        video_feat_root=str(video_feat_root),
        audio_ext=audio_ext,
        video_ext=video_ext,
        video_feat_key=video_feat_key,
        audio_input_dim=audio_input_dim,
        video_input_dim=video_input_dim,
        allow_missing_audio=allow_missing_audio,
        allow_missing_video=allow_missing_video,
    )
    ds_va = FusionAVNpyDataset(
        items_val,
        label2id=label2id,
        audio_feat_root=str(audio_feat_root),
        video_feat_root=str(video_feat_root),
        audio_ext=audio_ext,
        video_ext=video_ext,
        video_feat_key=video_feat_key,
        audio_input_dim=audio_input_dim,
        video_input_dim=video_input_dim,
        allow_missing_audio=allow_missing_audio,
        allow_missing_video=allow_missing_video,
    )
    ds_te = FusionAVNpyDataset(
        items_test,
        label2id=label2id,
        audio_feat_root=str(audio_feat_root),
        video_feat_root=str(video_feat_root),
        audio_ext=audio_ext,
        video_ext=video_ext,
        video_feat_key=video_feat_key,
        audio_input_dim=audio_input_dim,
        video_input_dim=video_input_dim,
        allow_missing_audio=allow_missing_audio,
        allow_missing_video=allow_missing_video,
    )

    collate_fn = partial(
        collate_fusion_av_npy,
        pad_mode=pad_mode,
        max_frames_audio=max_frames_audio,
        max_frames_video=max_frames_video,
    )

    pin_memory = bool(e.get("pin_memory", True))
    persistent_workers = bool(e.get("persistent_workers", True)) if num_workers > 0 else False

    tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return tr, va, te
