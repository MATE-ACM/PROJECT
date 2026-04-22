from __future__ import annotations

"""Expert registry and import side effects for model construction."""

from .registry import build_expert

from src.ser.experts.audio import audio_whisper
from src.ser.experts.audio import audio_whisper_experts
from src.ser.experts.audio import audio_WavLM_experts
from src.ser.experts.video import video_experts
from src.ser.experts.video import video_hsemotion_experts

# fusion experts
from src.ser.experts.fusion import fusion_av_xattn
from src.ser.experts.fusion import fusion_av_pool_mlp
from src.ser.experts.fusion import fusion_av_gated
from src.ser.experts.fusion import fusion_av_multlite
from src.ser.experts.fusion import fusion_av_lmf

from src.ser.experts.fusion import fusion_av_film
from src.ser.experts.fusion import fusion_av_tfnlite
from src.ser.experts.fusion import fusion_av_pool_mlp_meta

from src.ser.experts.fusion import fusion_avt_lmf
from src.ser.experts.fusion import fusion_avt_xattn
from src.ser.experts.fusion import fusion_avt_pool_mlp

from src.ser.experts.fusion import fusion_avt_merbench
from src.ser.experts.txt import txt_experts
__all__ = ["build_expert"]
