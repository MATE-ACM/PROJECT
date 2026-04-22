from __future__ import annotations

"""Adapter that exposes MERBench multimodal models through the local expert registry."""

from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.ser.experts.registry import register_expert
from src.ser.experts.merbench_toolkit.models import get_models

def _masked_mean_pool(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """x: [B,T,D], mask: [B,T] (1=valid,0=pad) -> [B,D]"""
    if x.ndim == 2:
        return x
    if mask is None:
        return x.mean(dim=1)
    m = mask.to(dtype=x.dtype)
    denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * m.unsqueeze(-1)).sum(dim=1) / denom

def _rightpad_to_leftpad(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """把右侧 padding 的序列转换成左侧 padding（保持时间顺序）。

    x: [B,T,D], mask: [B,T] with pattern [1..1,0..0]
    return: [B,T,D] with valid at the end.
    """
    if x.ndim != 3:
        return x
    if mask is None:
        return x

    B, T, D = x.shape
    lens = mask.sum(dim=1).to(dtype=torch.long)
    out = x.new_zeros((B, T, D))
    for i in range(B):
        L = int(lens[i].item())
        if L <= 0:
            continue
        if L >= T:
            out[i] = x[i]
        else:
            out[i, T - L : T] = x[i, :L]
    return out

@register_expert("fusion_avt_merbench")
class FusionAVTMerBench(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        if "expert" in cfg:
            e = (cfg.get("expert", {}) or {})
            train_cfg = (cfg.get("train", {}) or {})
        else:
            e = (cfg or {})
            train_cfg = {}

        self.model_name = str(e.get("mer_model", e.get("model", "attention")))
        self.feat_type = str(e.get("feat_type", "utt"))  # utt / frm_align / frm_unalign

        audio_dim = int(e.get("audio_input_dim"))
        video_dim = int(e.get("video_input_dim"))
        text_dim = int(e.get("text_input_dim"))

        num_classes = e.get("num_classes", None)
        if num_classes is None:
            num_classes = train_cfg.get("num_classes", None)
        if num_classes is None:
            raise KeyError("fusion_avt_merbench requires expert.num_classes (or train.num_classes)")
        num_classes = int(num_classes)

        # MERBench args
        args = SimpleNamespace(
            model=self.model_name,
            feat_type=self.feat_type if self.feat_type != "utt" else "utt",
            audio_dim=audio_dim,
            video_dim=video_dim,
            text_dim=text_dim,
            output_dim1=num_classes,
            output_dim2=0,
            hidden_dim=int(e.get("hidden_dim", 128)),
            dropout=float(e.get("dropout", 0.2)),
            grad_clip=float(e.get("grad_clip", 0.8)),
            # model-specific optional params (kept for YAML-compat)
            rank=int(e.get("rank", 4)),
            beta=float(e.get("beta", 0.01)),
            gamma=float(e.get("gamma", 0.01)),
            # MULT params
            text_out=int(e.get("text_out", 64)),
            audio_out=int(e.get("audio_out", 64)),
            video_out=int(e.get("video_out", 64)),
            d_model=int(e.get("d_model", 64)),
            n_heads=int(e.get("n_heads", 4)),
            n_layers=int(e.get("n_layers", 2)),
            attn_dropout=float(e.get("attn_dropout", 0.1)),
            relu_dropout=float(e.get("relu_dropout", 0.1)),
            res_dropout=float(e.get("res_dropout", 0.1)),
            out_dropout=float(e.get("out_dropout", 0.1)),
            embed_dropout=float(e.get("embed_dropout", 0.1)),
        )

        self.mer = get_models(args)
        self.num_classes = num_classes

    def _prep_inputs(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        xa = batch["x_audio"]
        xv = batch["x_video"]
        xt = batch["x_text"]
        ma = batch.get("x_audio_mask")
        mv = batch.get("x_video_mask")
        mt = batch.get("x_text_mask")

        # utt: pool to [B,D]
        if self.feat_type == "utt":
            xa = _masked_mean_pool(xa, ma)
            xv = _masked_mean_pool(xv, mv)
            xt = _masked_mean_pool(xt, mt)
        else:
            xa = _rightpad_to_leftpad(xa, ma)
            xv = _rightpad_to_leftpad(xv, mv)
            xt = _rightpad_to_leftpad(xt, mt)

        return {"audios": xa, "videos": xv, "texts": xt}

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        inputs = self._prep_inputs(batch)
        _, emos_out, _, _ = self.mer(inputs)
        return emos_out

    @torch.no_grad()
    def forward_with_extras(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self._prep_inputs(batch)
        feats, emos_out, _, interloss = self.mer(inputs)
        return emos_out, feats, interloss
