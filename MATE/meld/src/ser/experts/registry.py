from __future__ import annotations

"""Registry for expert model constructors."""

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
