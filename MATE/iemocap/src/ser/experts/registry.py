from __future__ import annotations

"""
【文件作用】专家注册表：通过 @register_expert("name") 注册，实现 build_expert(cfg) 工厂。

（如果你在 IDE 里阅读代码：先看本段，再往下看实现。）
"""

from typing import Dict, Type, Any

_EXPERTS: Dict[str, Type] = {}

def register_expert(name: str):
    def deco(cls):
        _EXPERTS[name] = cls
        return cls
    return deco

def build_expert(cfg: Dict[str, Any]):
    typ = cfg["type"]
    if typ not in _EXPERTS:
        raise KeyError(f"Unknown expert type: {typ}. Available: {list(_EXPERTS.keys())}")
    return _EXPERTS[typ](cfg)
