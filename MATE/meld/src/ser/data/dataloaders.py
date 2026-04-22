from __future__ import annotations

"""src.ser.data.dataloaders

Global dataloader dispatcher.

Prefix routing:
- audio_*  -> src.ser.data.audio.audio_dataloaders.make_audio_loaders
- video_*  -> src.ser.data.video.video_dataloaders.make_video_loaders
- txt_*    -> src.ser.data.txt.txt_dataloaders.make_txt_loaders
- fusion_* -> src.ser.data.fusion.fusion_dataloaders.make_fusion_loaders

This file should stay THIN and stable.
"""

from typing import Dict, Any, List

def make_loaders(
    items_train: List[Dict[str, Any]],
    items_val: List[Dict[str, Any]],
    items_test: List[Dict[str, Any]],
    label2id: Dict[str, int],
    expert_cfg: Dict[str, Any],
    expert_type: str,
    batch_size: int = 32,
    num_workers: int = 4,
):
    etype = str(expert_type).strip()

    if etype.startswith("audio_"):
        from src.ser.data.audio.audio_dataloaders import make_audio_loaders

        return make_audio_loaders(
            items_train=items_train,
            items_val=items_val,
            items_test=items_test,
            label2id=label2id,
            cfg=expert_cfg,
            expert_type=etype,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    if etype.startswith("video_"):
        from src.ser.data.video.video_dataloaders import make_video_loaders

        return make_video_loaders(
            items_train=items_train,
            items_val=items_val,
            items_test=items_test,
            label2id=label2id,
            cfg=expert_cfg,
            expert_type=etype,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    if etype.startswith("txt_") or etype.startswith("text_"):
        from src.ser.data.txt.txt_dataloaders import make_txt_loaders

        return make_txt_loaders(
            items_train=items_train,
            items_val=items_val,
            items_test=items_test,
            label2id=label2id,
            cfg=expert_cfg,
            expert_type=etype,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    if etype.startswith("fusion_"):
        from src.ser.data.fusion.fusion_dataloaders import make_fusion_loaders

        return make_fusion_loaders(
            items_train=items_train,
            items_val=items_val,
            items_test=items_test,
            label2id=label2id,
            cfg=expert_cfg,
            expert_type=etype,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    raise KeyError(f"fusion_.")
