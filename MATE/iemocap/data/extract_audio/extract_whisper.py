#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from __future__ import annotations
import os
import json
import gc
import math
import traceback
import subprocess
from pathlib import Path
from typing import Dict, Any, Iterator, Optional, Tuple, List

import numpy as np
from tqdm import tqdm

import torch
import torchaudio
from transformers import (
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperModel,
    WhisperForConditionalGeneration,
)

# 国内可选：不需要就删
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# =========================

# =========================
CFG = {
    # 你的 IEMOCAP manifest（4-way 或 6-way 都行）
    "manifest_jsonl": r"patch_missing_by_feature_audio.jsonl",

    # 如果发现 rec["audio_path"] 不存在，且 rec 提供 dialog_* + start/end，则会自动切分到这个目录
    "auto_segment_if_missing": True,
    "emit_audio_root": r"iemocap_utt_audio_16k",

    # 输出根目录（会自动创建多个子目录）
    "out_root": r"feature",

    # 运行模式： "acoustic" | "semantic" | "both"
    "mode": "both",


    "whisper_id": "whisper-large-v3",

    # 音频处理
    "target_sr": 16000,
    "max_sec": 30.0,      # whisper 输入默认 30s；>0 则截断到 max_sec
    "min_samples": 800,   # 太短则输出全 0

    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_fp16": True,     # cuda 上建议 True；cpu 自动变 float32

    # 输出控制
    "save_frame_level": True,  # 保存 enc_frame / dec_frame（可变长）
    "save_pooled": True,       # 保存 enc_mean / dec_mean（固定长度）
    "pooling": "mean",         # 目前只做 mean

    # semantic 模式：ASR 配置
    "asr_language": "en",      # IEMOCAP 是英文；如需自动检测可设为 None（但更慢/不稳定）
    "asr_task": "transcribe",
    "gen_max_new_tokens": 128,
    "gen_num_beams": 1,

    # 是否保存 ASR 文本（只在 semantic/both 生效）
    "save_asr_text": True,

    # 断点续跑
    "resume": True,

    # 失败日志
    "fail_jsonl": "_failures_whisper_expert.jsonl",
}
# =========================


# -------------------------
# Utils
# -------------------------
def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def pick_audio_source(rec: Dict[str, Any]) -> Optional[str]:
    # 优先对话 wav，再对话视频（ffmpeg 也能从 avi/mp4 抠音频）
    for k in ["dialog_audio_path", "dialog_wav_path", "dialog_audio", "dialog_video_path", "dialog_avi_path", "video_path"]:
        v = rec.get(k)
        if v and os.path.exists(str(v)):
            return str(v)
    # 其次如果 audio_path 自己就存在
    ap = rec.get("audio_path")
    if ap and os.path.exists(str(ap)):
        return str(ap)
    return None


def ffmpeg_segment_wav(src: str, st: float, ed: float, out_wav: str, target_sr: int = 16000):
    ensure_dir(Path(out_wav).parent)
    dur = max(0.01, float(ed) - float(st))
    # -ss 放前面更快；一般 utt 级够用了
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{st:.3f}",
        "-t", f"{dur:.3f}",
        "-i", src,
        "-vn",
        "-ac", "1",
        "-ar", str(int(target_sr)),
        out_wav,
    ]
    subprocess.run(cmd, check=True)


def load_audio_1d(wav_path: str, target_sr: int, max_sec: float) -> Optional[torch.Tensor]:
    try:
        wav, sr = torchaudio.load(wav_path)  # [C, T]
        if wav.numel() == 0:
            return None
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)

        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)

        if max_sec and max_sec > 0:
            max_len = int(target_sr * float(max_sec))
            if wav.numel() > max_len:
                wav = wav[:max_len]

        return wav.contiguous()
    except Exception:
        return None


def npy_ok(path: str) -> bool:
    try:
        _ = np.load(path, allow_pickle=False)
        return True
    except Exception:
        return False


def cleanup_model(obj):
    try:
        del obj
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -------------------------
# Whisper Expert
# -------------------------
class WhisperExpert:
    """
    一个类同时支持：
    - acoustic：WhisperModel.encoder hidden states
    - semantic：WhisperForConditionalGeneration generate + decoder hidden states
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = cfg["device"]
        self.target_sr = int(cfg["target_sr"])
        self.max_sec = float(cfg["max_sec"])
        self.min_samples = int(cfg["min_samples"])

        ensure_dir(cfg["out_root"])
        self.out_root = Path(cfg["out_root"])


        self.dir_enc_frame = self.out_root / "whisper_enc_frame"
        self.dir_enc_mean  = self.out_root / "whisper_enc_mean"
        self.dir_dec_frame = self.out_root / "whisper_dec_frame"
        self.dir_dec_mean  = self.out_root / "whisper_dec_mean"
        self.dir_tokens    = self.out_root / "whisper_asr_tokens"
        self.asr_text_path = self.out_root / "whisper_asr_text.jsonl"

        if cfg["save_frame_level"]:
            ensure_dir(self.dir_enc_frame)
            ensure_dir(self.dir_dec_frame)
        if cfg["save_pooled"]:
            ensure_dir(self.dir_enc_mean)
            ensure_dir(self.dir_dec_mean)
        if cfg["save_asr_text"]:
            ensure_dir(self.asr_text_path.parent)
        ensure_dir(self.dir_tokens)

        model_id = cfg["whisper_id"]
        use_fp16 = bool(cfg["use_fp16"]) and (self.device != "cpu") and torch.cuda.is_available()
        self.dtype = torch.float16 if use_fp16 else torch.float32

        mode = cfg["mode"].lower()
        assert mode in ["acoustic", "semantic", "both"]

        # acoustic：用 WhisperFeatureExtractor + WhisperModel（只 encoder）
        if mode in ["acoustic", "both"]:
            print(f"[INFO] Loading WhisperModel (encoder) {model_id} on {self.device} dtype={self.dtype}.")
            self.feat_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
            self.enc_model = WhisperModel.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device).eval()
        else:
            self.feat_extractor = None
            self.enc_model = None

        # semantic：用 WhisperProcessor + WhisperForConditionalGeneration（generate + decoder hs）
        if mode in ["semantic", "both"]:
            print(f"[INFO] Loading WhisperForConditionalGeneration (ASR) {model_id} on {self.device} dtype={self.dtype}.")
            self.processor = WhisperProcessor.from_pretrained(model_id)
            self.asr_model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device).eval()
        else:
            self.processor = None
            self.asr_model = None

        self.failures: List[Dict[str, Any]] = []
        self.asr_text_buf: List[Dict[str, Any]] = []

    @torch.no_grad()
    def extract_encoder(self, wav_1d: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns:
          enc_frame: [T, D]
          enc_mean : [D]
        """
        if wav_1d.numel() < self.min_samples:
            # dim 先用模型推断不方便；用 1x1 占位，后面保存前再修正
            return np.zeros((1, 1), np.float32), np.zeros((1,), np.float32)

        # 真实长度（用于裁 padding）
        real_len_sec = wav_1d.numel() / float(self.target_sr)

        inputs = self.feat_extractor(
            wav_1d.cpu().numpy(),
            sampling_rate=self.target_sr,
            return_tensors="pt"
        )
        feats = inputs.input_features.to(self.device)
        if self.dtype == torch.float16:
            feats = feats.half()

        out = self.enc_model.encoder(feats)
        hs = out.last_hidden_state.squeeze(0)  # [T, D]

        # Whisper encoder 大致 20ms/frame（与你 ravdess 的写法一致）
        valid_frames = int(real_len_sec / 0.02)
        valid_frames = min(valid_frames + 2, hs.shape[0])
        valid_frames = max(1, valid_frames)

        hs = hs[:valid_frames].float()  # 转 float32 方便保存
        enc_frame = hs.cpu().numpy().astype(np.float32)
        enc_mean = hs.mean(dim=0).cpu().numpy().astype(np.float32)
        return enc_frame, enc_mean

    @torch.no_grad()
    def extract_decoder_semantic(self, wav_1d: torch.Tensor, utt_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Semantic baseline:
          - generate tokens
          - forward w/ decoder_input_ids to obtain decoder last-layer hidden states

        returns:
          dec_frame: [L, D]  (token-level)
          dec_mean : [D]
          tokens   : [L] int32
          text     : decoded transcript
        """
        if wav_1d.numel() < self.min_samples:
            return np.zeros((1, 1), np.float32), np.zeros((1,), np.float32), np.zeros((1,), np.int32), ""

        inputs = self.processor(
            wav_1d.cpu().numpy(),
            sampling_rate=self.target_sr,
            return_tensors="pt",
        )
        feats = inputs.input_features.to(self.device)
        if self.dtype == torch.float16:
            feats = feats.half()

        # 固定语言/任务（更稳定）
        forced_ids = None
        lang = self.cfg.get("asr_language")
        task = self.cfg.get("asr_task", "transcribe")
        if lang:
            forced_ids = self.processor.get_decoder_prompt_ids(language=lang, task=task)

        gen = self.asr_model.generate(
            inputs=feats,
            forced_decoder_ids=forced_ids,
            max_new_tokens=int(self.cfg["gen_max_new_tokens"]),
            num_beams=int(self.cfg["gen_num_beams"]),
            return_dict_in_generate=True,
            output_scores=False,
        )
        seq = gen.sequences  # [1, L]
        tokens = seq.squeeze(0).detach().cpu().numpy().astype(np.int32)

        text = self.processor.batch_decode(seq, skip_special_tokens=True)[0]

        # 再跑一遍 forward 拿 decoder hidden states
        out = self.asr_model(
            input_features=feats,
            decoder_input_ids=seq,
            output_hidden_states=True,
            return_dict=True,
        )
        # decoder_hidden_states: tuple(len=L_layers+1?) 最后一项是 last layer output
        dhs = out.decoder_hidden_states[-1].squeeze(0).float()  # [L, D]
        dec_frame = dhs.cpu().numpy().astype(np.float32)
        dec_mean = dhs.mean(dim=0).cpu().numpy().astype(np.float32)
        return dec_frame, dec_mean, tokens, text

    def save_asr_text(self):
        if not self.cfg.get("save_asr_text"):
            return
        if not self.asr_text_buf:
            return
        # 追加写入（方便断点续跑）
        with open(self.asr_text_path, "a", encoding="utf-8") as f:
            for r in self.asr_text_buf:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.asr_text_buf.clear()

    def dump_failures(self):
        if not self.failures:
            return
        p = self.out_root / self.cfg["fail_jsonl"]
        with open(p, "w", encoding="utf-8") as f:
            for r in self.failures:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[WARN] failures saved -> {p} (n={len(self.failures)})")


# -------------------------
# Main runner
# -------------------------
def main():
    cfg = CFG
    ensure_dir(cfg["out_root"])
    mode = cfg["mode"].lower()

    expert = WhisperExpert(cfg)

    # 统计
    n_total = 0
    n_ok = 0
    n_skip = 0

    # ASR text 断点续跑：如果文件已存在，先统计已有 utt_id，避免重复写
    done_text = set()
    if cfg["save_asr_text"] and expert.asr_text_path.exists() and cfg["resume"]:
        try:
            with open(expert.asr_text_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if "utt_id" in r:
                        done_text.add(str(r["utt_id"]))
        except Exception:
            done_text = set()

    for rec in tqdm(iter_jsonl(cfg["manifest_jsonl"]), desc=f"whisper_{mode}"):
        n_total += 1
        utt_id = str(rec.get("utt_id", "")).strip()
        if not utt_id:
            continue

        # 1) 准备 wav_path（优先 rec["audio_path"]；否则自动切分）
        wav_path = str(rec.get("audio_path", "")).strip()
        if wav_path and os.path.exists(wav_path):
            pass
        else:
            if not cfg["auto_segment_if_missing"]:
                expert.failures.append({"utt_id": utt_id, "err": "audio_path_missing", "audio_path": wav_path})
                continue

            src = pick_audio_source(rec)
            if not src:
                expert.failures.append({"utt_id": utt_id, "err": "no_audio_source", "rec_keys": list(rec.keys())})
                continue

            st = safe_float(rec.get("start_sec", 0.0), 0.0)
            ed = safe_float(rec.get("end_sec", 0.0), 0.0)
            if ed <= st:
                expert.failures.append({"utt_id": utt_id, "err": f"bad_time_range st={st} ed={ed}"})
                continue

            wav_path = str(Path(cfg["emit_audio_root"]) / f"{utt_id}.wav")
            if cfg["resume"] and os.path.exists(wav_path):
                pass
            else:
                try:
                    ffmpeg_segment_wav(src, st, ed, wav_path, target_sr=int(cfg["target_sr"]))
                except Exception as e:
                    expert.failures.append({"utt_id": utt_id, "err": f"ffmpeg_failed: {e}", "src": src})
                    continue

        # 2) 读取音频
        wav_1d = load_audio_1d(wav_path, int(cfg["target_sr"]), float(cfg["max_sec"]))
        if wav_1d is None:
            expert.failures.append({"utt_id": utt_id, "err": "load_audio_failed", "wav_path": wav_path})
            continue

        # 3) 输出路径（按 utt_id）
        p_enc_frame = expert.dir_enc_frame / f"{utt_id}.npy"
        p_enc_mean  = expert.dir_enc_mean  / f"{utt_id}.npy"
        p_dec_frame = expert.dir_dec_frame / f"{utt_id}.npy"
        p_dec_mean  = expert.dir_dec_mean  / f"{utt_id}.npy"
        p_tokens    = expert.dir_tokens    / f"{utt_id}.npy"

        # resume 跳过逻辑：按模式检查目标文件是否存在且可读
        def _need(path: Path) -> bool:
            return (not cfg["resume"]) or (not path.exists()) or (not npy_ok(str(path)))

        need_any = False
        if mode in ["acoustic", "both"]:
            if cfg["save_frame_level"]:
                need_any |= _need(p_enc_frame)
            if cfg["save_pooled"]:
                need_any |= _need(p_enc_mean)
        if mode in ["semantic", "both"]:
            if cfg["save_frame_level"]:
                need_any |= _need(p_dec_frame)
            if cfg["save_pooled"]:
                need_any |= _need(p_dec_mean)
            need_any |= _need(p_tokens)
            if cfg["save_asr_text"]:
                need_any |= (utt_id not in done_text)

        if not need_any:
            n_skip += 1
            continue

        try:
            # acoustic encoder
            if mode in ["acoustic", "both"]:
                enc_frame, enc_mean = expert.extract_encoder(wav_1d)

                # 如果是太短返回的 1x1 占位，补成正确 dim（用模型 hidden_size）
                if enc_frame.shape[1] == 1 and expert.enc_model is not None:
                    d = int(expert.enc_model.config.d_model)
                    enc_frame = np.zeros((1, d), np.float32)
                    enc_mean = np.zeros((d,), np.float32)

                if cfg["save_frame_level"] and _need(p_enc_frame):
                    ensure_dir(p_enc_frame.parent)
                    np.save(p_enc_frame, enc_frame)
                if cfg["save_pooled"] and _need(p_enc_mean):
                    ensure_dir(p_enc_mean.parent)
                    np.save(p_enc_mean, enc_mean)

            # semantic decoder
            if mode in ["semantic", "both"]:
                dec_frame, dec_mean, tokens, text = expert.extract_decoder_semantic(wav_1d, utt_id)

                if dec_frame.shape[1] == 1 and expert.asr_model is not None:
                    d = int(expert.asr_model.model.config.d_model)
                    dec_frame = np.zeros((1, d), np.float32)
                    dec_mean = np.zeros((d,), np.float32)

                if cfg["save_frame_level"] and _need(p_dec_frame):
                    ensure_dir(p_dec_frame.parent)
                    np.save(p_dec_frame, dec_frame)
                if cfg["save_pooled"] and _need(p_dec_mean):
                    ensure_dir(p_dec_mean.parent)
                    np.save(p_dec_mean, dec_mean)
                if _need(p_tokens):
                    ensure_dir(p_tokens.parent)
                    np.save(p_tokens, tokens)

                if cfg["save_asr_text"] and (utt_id not in done_text):
                    expert.asr_text_buf.append({"utt_id": utt_id, "text": text})
                    done_text.add(utt_id)
                    # 每 200 条 flush 一次
                    if len(expert.asr_text_buf) >= 200:
                        expert.save_asr_text()

            n_ok += 1

        except KeyboardInterrupt:
            raise
        except Exception as e:
            expert.failures.append({"utt_id": utt_id, "err": str(e), "trace": traceback.format_exc(), "wav_path": wav_path})
            continue

    # flush
    expert.save_asr_text()
    expert.dump_failures()

    print("[DONE]")
    print(f"  manifest = {cfg['manifest_jsonl']}")
    print(f"  mode     = {mode}")
    print(f"  out_root = {cfg['out_root']}")
    print(f"  total={n_total} ok={n_ok} skipped={n_skip} failures={len(expert.failures)}")

    cleanup_model(expert)


if __name__ == "__main__":
    main()
