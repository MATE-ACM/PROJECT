from __future__ import annotations

"""Text experts.

Add new text experts here and import them in src.ser.experts.__init__
so that @register_expert is executed.
"""

from . import txt_experts  # noqa: F401

__all__ = ["txt_experts"]
