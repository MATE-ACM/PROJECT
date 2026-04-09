from __future__ import annotations

"""
【文件作用】通用训练循环与工具：优化器/调度器/早停/保存 best checkpoint 等。


"""

from typing import Dict, Any, Tuple
import math
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def make_scheduler(opt, scheduler: str, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int):
        if step < warmup_steps and warmup_steps > 0:
            return (step + 1) / warmup_steps
        # cosine decay to 0.1
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if scheduler == "cosine":
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return LambdaLR(opt, lr_lambda=lr_lambda)
