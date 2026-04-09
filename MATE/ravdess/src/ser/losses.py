from __future__ import annotations

"""
【文件作用】训练损失：CE/Weighted CE/Focal/Label Smoothing 等。

（如果你在 IDE 里阅读代码：先看本段，再往下看实现。）
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_loss(loss_cfg: Dict[str, Any], class_weights: torch.Tensor | None):
    typ = loss_cfg.get("type", "ce")
    if typ == "ce":
        return nn.CrossEntropyLoss()
    if typ == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if typ == "label_smoothing":
        eps = float(loss_cfg.get("label_smoothing", 0.05))
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=eps)
    if typ == "focal":
        gamma = float(loss_cfg.get("focal_gamma", 1.5))
        def focal_loss(logits, y):
            ce = F.cross_entropy(logits, y, weight=class_weights, reduction="none")
            p = torch.softmax(logits, dim=-1).gather(1, y.view(-1,1)).squeeze(1).clamp_min(1e-6)
            loss = ((1 - p) ** gamma) * ce
            return loss.mean()
        return focal_loss
    raise KeyError(f"Unknown loss type: {typ}")
