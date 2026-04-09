from __future__ import annotations

"""src.ser.experts

专家注册与构建层：
- 新增专家时：
  1) 新建一个 py 文件（例如 audio_whisper_npy.py）
  2) 用 @register_expert("your_expert_type") 注册
  3) 在本 __init__.py 里 import 该模块（让注册代码执行）
  4) 写 configs/experts/*.yaml
"""

from .registry import build_expert


from src.ser.experts.audio import audio_whisper  # noqa: F401  注册: audio_whisper_npy
from src.ser.experts.audio import audio_whisper_experts
from src.ser.experts.audio import audio_WavLM_experts
from src.ser.experts.video import video_experts  # noqa: F401  注册: video_experts_npy
from src.ser.experts.video import video_hsemotion_experts

# fusion experts
from src.ser.experts.fusion import fusion_av_xattn  # noqa: F401  注册: fusion_av_xattn
from src.ser.experts.fusion import fusion_av_pool_mlp
from src.ser.experts.fusion import fusion_av_gated
from src.ser.experts.fusion import fusion_av_multlite
from src.ser.experts.fusion import fusion_av_lmf

from src.ser.experts.fusion import fusion_av_film
from src.ser.experts.fusion import fusion_av_tfnlite
from src.ser.experts.fusion import fusion_av_pool_mlp_meta
__all__ = ["build_expert"]
