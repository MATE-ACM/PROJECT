from __future__ import annotations

"""
【文件作用】模块说明：请阅读本文件顶部注释。

（如果你在 IDE 里阅读代码：先看本段，再往下看实现。）
"""

from typing import Dict, Any
import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
