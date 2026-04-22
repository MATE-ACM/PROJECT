from __future__ import annotations

from functools import partial
from typing import Any, Dict, List

from torch.utils.data import DataLoader

def make_txt_loaders(
    items_train: List[Dict[str, Any]],
    items_val: List[Dict[str, Any]],
    items_test: List[Dict[str, Any]],
    label2id: Dict[str, int],
    cfg: Dict[str, Any],
    expert_type: str,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Create dataloaders for text modality.

    The config lookup follows the same rule as audio/video:
      - prefer expert.feat_root
      - fallback to data.feat_root
    """

    e = cfg.get("expert", cfg)

    from src.ser.data.txt.txt_npy import TxtNpyDataset, collate_txt_npy

    feat_root = e.get("feat_root") or (cfg.get("data", {}) or {}).get("feat_root")
    if feat_root is None:
        raise KeyError(f"[{expert_type}] missing feat_root. Put it in expert.feat_root (recommended) or data.feat_root.")

    pad_mode = str(e.get("pad_mode", "right"))

    max_frames = e.get("max_frames", None)
    max_frames = int(max_frames) if max_frames is not None else None

    input_dim = e.get("input_dim")
    input_dim = int(input_dim) if input_dim is not None else None

    ext = str(e.get("ext", ".npy"))
    feat_key = str(e.get("feat_key", "x"))
    mmap = bool(e.get("mmap", False))

    ds_tr = TxtNpyDataset(
        items_train,
        label2id=label2id,
        feat_root=str(feat_root),
        ext=ext,
        feat_key=feat_key,
        pad_mode=pad_mode,
        max_frames=max_frames,
        mmap=mmap,
        input_dim=input_dim,
    )
    ds_va = TxtNpyDataset(
        items_val,
        label2id=label2id,
        feat_root=str(feat_root),
        ext=ext,
        feat_key=feat_key,
        pad_mode=pad_mode,
        max_frames=max_frames,
        mmap=mmap,
        input_dim=input_dim,
    )
    ds_te = TxtNpyDataset(
        items_test,
        label2id=label2id,
        feat_root=str(feat_root),
        ext=ext,
        feat_key=feat_key,
        pad_mode=pad_mode,
        max_frames=max_frames,
        mmap=mmap,
        input_dim=input_dim,
    )

    collate_fn = partial(collate_txt_npy, pad_mode=pad_mode, max_frames=max_frames)

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
