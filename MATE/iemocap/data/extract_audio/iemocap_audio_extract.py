#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



  OUTPUT_ROOT/
    audio_wavlm_avg_all/*.npy
    audio_wavlm_avg_last12/*.npy
    audio_whisper/*.npy
    _failures_*.jsonl / *_metadata.csv


"""

from __future__ import annotations

import os
import gc
import json
import math
import logging
import traceback
import subprocess
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio
from transformers import (
    WavLMModel, Wav2Vec2FeatureExtractor,
    WhisperModel, WhisperFeatureExtractor
)


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ================= ⚙️ 配置区域（只改这里） =================
CONFIG = {
    # IEMOCAP 6way / 4way jsonl（也支持 csv）
    
    "MANIFEST_PATH": r"patch_missing_by_feature_audio.jsonl",

    # 如果还没切 utt wav，把它开成 True
    "RUN_SEGMENT_WAV": True,
    "FFMPEG": r"ffmpeg.exe",
    "OVERWRITE_SEGMENT": True,

   
    "EMIT_AUDIO_ROOT": r"iemocap_utt_audio_16k",

  
    "OUTPUT_ROOT": r"feature",

    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # 开关
    "RUN_WAVLM_LAYERAVG": True,
    "RUN_WHISPER": True,

    # 音频预处理
    "TARGET_SR": 16000,
    "MAX_SEC": 30.0,       # >0 则截断；0 则不截断
    "MIN_SAMPLES": 800,    # 太短则输出全 0

 
    "DO_ZMUV": False,

    # 模型
    "WAVLM_ID": "wavlm-large",
    "WHISPER_ID": "whisper-large-v3",

    # WavLM layer-avg 输出控制
    "EXPORT_AVG_ALL": True,
    "EXPORT_AVG_LAST12": True,
    "LAST_N": 12,

    # 若已存在 npy 是否跳过
    "RESUME": True,

    # 日志
    "LOG_PATH": "iemocap_audio_extract.log",
}
# ============================================================


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def npy_ok(p: str) -> bool:
    try:
        arr = np.load(p, mmap_mode="r")
        return (arr.ndim >= 2) and (arr.shape[0] >= 1) and (arr.shape[1] >= 1)
    except Exception:
        return False


def read_manifest_to_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)

    # 对齐 id：utt_id -> uid（兼容你 RAVDESS 脚本）
    if "utt_id" in df.columns and "uid" not in df.columns:
        df = df.rename(columns={"utt_id": "uid"})
    return df


def pick_audio_source(row: Any) -> Optional[str]:

    for k in ("dialog_audio_path", "dialog_video_path", "audio_src", "video_src"):
        if hasattr(row, k):
            v = getattr(row, k)
            if isinstance(v, str) and v:
                return v
    return None


def ffmpeg_segment_wav(src: str, start_sec: float, end_sec: float, out_wav: str) -> bool:
    """
    ffmpeg -ss <start> -to <end> -i <src> -ac 1 -ar 16000 -vn <out.wav>
    """
    ensure_dir(os.path.dirname(out_wav))
    cmd = [
        CONFIG["FFMPEG"],
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{start_sec:.3f}",
        "-to", f"{end_sec:.3f}",
        "-i", src,
        "-ac", "1",
        "-ar", str(CONFIG["TARGET_SR"]),
        "-vn",
        out_wav,
        "-y",
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def run_segment_wavs(df: pd.DataFrame) -> None:
    if "audio_path" not in df.columns:
        raise ValueError("Manifest missing required column: audio_path")

    need_cols = {"start_sec", "end_sec"}
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"RUN_SEGMENT_WAV=True but manifest missing column: {c}")

    failures = []
    done = 0
    skip = 0

    for r in tqdm(df.itertuples(index=False), total=len(df), desc="segment_wav"):
        uid = str(getattr(r, "uid"))
        out_wav = getattr(r, "audio_path")
        if not isinstance(out_wav, str) or not out_wav:
            out_wav = os.path.join(CONFIG["EMIT_AUDIO_ROOT"], f"{uid}.wav")

        if (not CONFIG["OVERWRITE_SEGMENT"]) and os.path.exists(out_wav):
            skip += 1
            continue

        src = pick_audio_source(r)
        if not src or (not os.path.exists(src)):
            failures.append({"uid": uid, "err": "missing_audio_source", "src": src})
            continue

        st = float(getattr(r, "start_sec", 0.0))
        ed = float(getattr(r, "end_sec", 0.0))
        if ed <= st:
            failures.append({"uid": uid, "err": "bad_time", "start_sec": st, "end_sec": ed})
            continue

        ok = ffmpeg_segment_wav(src, st, ed, out_wav)
        if ok:
            done += 1
        else:
            failures.append({"uid": uid, "err": "ffmpeg_failed", "src": src})

    print(f"[segment] done={done} skip={skip} failures={len(failures)}")
    if failures:
        p = os.path.join(CONFIG["OUTPUT_ROOT"], "_failures_segment_wav.jsonl")
        ensure_dir(CONFIG["OUTPUT_ROOT"])
        with open(p, "w", encoding="utf-8") as f:
            for x in failures:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"[segment] failures -> {p}")


def load_audio_1d(wav_path: str, target_sr: int) -> Optional[torch.Tensor]:
    """
    统一预处理：读 -> 单声道 -> 重采样 -> 截断 -> 去 DC -> (可选) ZMUV
    返回：1D tensor [T]
    """
    try:
        wav, sr = torchaudio.load(wav_path)  # [C, T]
        if wav.numel() == 0:
            return None

        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)  # [T]

        # resample
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            wav = resampler(wav)

        # truncate
        max_sec = float(CONFIG["MAX_SEC"])
        if max_sec > 0:
            max_len = int(max_sec * target_sr)
            if wav.numel() > max_len:
                wav = wav[:max_len]

        # DC removal
        wav = wav - wav.mean()

        # optional ZMUV
        if CONFIG["DO_ZMUV"]:
            std = wav.std()
            if std > 1e-7:
                wav = wav / std

        return wav.contiguous()
    except Exception:
        logging.error(traceback.format_exc())
        return None


# ===================== 🧠 WavLM：avg_all + avg_last12 =====================

class WavLMLayerAvgExtractor:
    """
    一次 forward，导出：
      - avg_all（所有 transformer 层平均，跳过 embedding）
      - avg_last12（最后12层平均，跳过 embedding）
    """
    def __init__(self, device: str):
        self.device = device
        self.target_sr = int(CONFIG["TARGET_SR"])

        self.export_all = bool(CONFIG["EXPORT_AVG_ALL"])
        self.export_last12 = bool(CONFIG["EXPORT_AVG_LAST12"])
        self.last_n = int(CONFIG["LAST_N"])

        assert self.export_all or self.export_last12, "至少导出一种 WavLM 特征。"

        self.dir_all = os.path.join(CONFIG["OUTPUT_ROOT"], "audio_wavlm_avg_all")
        self.dir_last12 = os.path.join(CONFIG["OUTPUT_ROOT"], "audio_wavlm_avg_last12")
        if self.export_all:
            ensure_dir(self.dir_all)
        if self.export_last12:
            ensure_dir(self.dir_last12)

        self.metadata_all: List[dict] = []
        self.metadata_last12: List[dict] = []
        self.failures: List[dict] = []

        print("Loading WavLM (output_hidden_states=True)...")
        model_id = CONFIG["WAVLM_ID"]
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

        dtype = torch.float16 if (device != "cpu" and torch.cuda.is_available()) else torch.float32
        self.model = WavLMModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        self.model.config.output_hidden_states = True

        self.hidden_size = int(getattr(self.model.config, "hidden_size", 1024))
        print(f"WavLM Loaded. hidden={self.hidden_size} dtype={self.model.dtype} device={device}")

    @torch.no_grad()
    def extract(self, wav_path: str) -> Optional[Dict[str, np.ndarray]]:
        wav_1d = load_audio_1d(wav_path, self.target_sr)
        if wav_1d is None:
            return None

        if wav_1d.numel() < int(CONFIG["MIN_SAMPLES"]):
            out = {}
            if self.export_all:
                out["avg_all"] = np.zeros((1, self.hidden_size), dtype=np.float32)
            if self.export_last12:
                out["avg_last12"] = np.zeros((1, self.hidden_size), dtype=np.float32)
            return out

        inputs = self.processor(wav_1d.cpu().numpy(), sampling_rate=self.target_sr, return_tensors="pt")
        x = inputs.input_values.to(self.device)
        if self.model.dtype == torch.float16:
            x = x.half()

        out = self.model(x)
        hs = out.hidden_states  # (emb, layer1..layerL)
        if hs is None or len(hs) <= 1:
            feat = out.last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)
            # 兜底：当作 avg_all
            return {"avg_all": feat} if self.export_all else {"avg_last12": feat}

        layers = hs[1:]  # skip embedding

        results: Dict[str, np.ndarray] = {}

        if self.export_all:
            acc = None
            for h in layers:
                h32 = h.float()
                acc = h32 if acc is None else (acc + h32)
            mean_all = (acc / float(len(layers))).squeeze(0)  # [T, C]
            results["avg_all"] = mean_all.cpu().numpy().astype(np.float32)

        if self.export_last12:
            sel = layers[-self.last_n:] if len(layers) >= self.last_n else layers
            acc = None
            for h in sel:
                h32 = h.float()
                acc = h32 if acc is None else (acc + h32)
            mean_last = (acc / float(len(sel))).squeeze(0)
            results["avg_last12"] = mean_last.cpu().numpy().astype(np.float32)

        return results

    def save_metadata(self):
        if self.export_all and self.metadata_all:
            p = os.path.join(self.dir_all, "_metadata.csv")
            pd.DataFrame(self.metadata_all).to_csv(p, index=False)
            print(f" [wavlm avg_all] metadata -> {p}")
        if self.export_last12 and self.metadata_last12:
            p = os.path.join(self.dir_last12, "_metadata.csv")
            pd.DataFrame(self.metadata_last12).to_csv(p, index=False)
            print(f"[wavlm avg_last12] metadata -> {p}")
        if self.failures:
            p = os.path.join(CONFIG["OUTPUT_ROOT"], "_failures_wavlm.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for r in self.failures:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[wavlm] failures {len(self.failures)} -> {p}")


# ===================== 🧠 Whisper：encoder hidden states =====================

class WhisperEncoderExtractor:
    def __init__(self, device: str):
        self.name = "audio_whisper"
        self.device = device
        self.target_sr = int(CONFIG["TARGET_SR"])
        self.save_dir = os.path.join(CONFIG["OUTPUT_ROOT"], self.name)
        ensure_dir(self.save_dir)

        self.metadata: List[dict] = []
        self.failures: List[dict] = []

        print(" Loading Whisper encoder...")
        model_id = CONFIG["WHISPER_ID"]
        self.processor = WhisperFeatureExtractor.from_pretrained(model_id)
        dtype = torch.float16 if (device != "cpu" and torch.cuda.is_available()) else torch.float32
        self.model = WhisperModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        self.hidden_size = int(getattr(self.model.config, "d_model", 1280))
        print(f" Whisper Loaded. hidden={self.hidden_size} dtype={self.model.dtype} device={device}")

    @torch.no_grad()
    def extract(self, wav_path: str) -> Optional[np.ndarray]:
        wav_1d = load_audio_1d(wav_path, self.target_sr)
        if wav_1d is None:
            return None

        if wav_1d.numel() < int(CONFIG["MIN_SAMPLES"]):
            return np.zeros((1, self.hidden_size), dtype=np.float32)

        real_len_sec = wav_1d.numel() / float(self.target_sr)

        inputs = self.processor(wav_1d.cpu().numpy(), sampling_rate=self.target_sr, return_tensors="pt")
        feats = inputs.input_features.to(self.device)
        if self.model.dtype == torch.float16:
            feats = feats.half()

        out = self.model.encoder(feats)
        hs = out.last_hidden_state.squeeze(0).cpu().numpy()  # [T, D]

        # Whisper 特征帧大约 20ms/帧（经验裁剪，避免 padding）
        valid_frames = int(real_len_sec / 0.02)
        valid_frames = min(valid_frames + 2, hs.shape[0])
        valid_frames = max(1, valid_frames)

        return hs[:valid_frames].astype(np.float32)

    def save_metadata(self):
        if self.metadata:
            p = os.path.join(self.save_dir, "_metadata.csv")
            pd.DataFrame(self.metadata).to_csv(p, index=False)
            print(f"[whisper] metadata -> {p}")
        if self.failures:
            p = os.path.join(self.save_dir, "_failures_whisper.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for r in self.failures:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f" [whisper] failures {len(self.failures)} -> {p}")


def cleanup_obj(obj):
    try:
        if hasattr(obj, "model"):
            del obj.model
        if hasattr(obj, "processor"):
            del obj.processor
    except Exception:
        pass
    try:
        del obj
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_wavlm_layeravg(ext: WavLMLayerAvgExtractor, df: pd.DataFrame):
    if "audio_path" not in df.columns:
        raise ValueError("Manifest missing required column: audio_path")

    print(" Start Task: WavLM layer-avg (avg_all + avg_last12)")
    skipped = 0
    processed = 0

    for r in tqdm(df.itertuples(index=False), total=len(df), desc="wavlm_layeravg"):
        uid = str(getattr(r, "uid"))
        wav_path = getattr(r, "audio_path")
        if not isinstance(wav_path, str) or not wav_path:
            wav_path = os.path.join(CONFIG["EMIT_AUDIO_ROOT"], f"{uid}.wav")

        need_all = ext.export_all
        need_last12 = ext.export_last12

        path_all = os.path.join(ext.dir_all, f"{uid}.npy") if need_all else None
        path_last12 = os.path.join(ext.dir_last12, f"{uid}.npy") if need_last12 else None

        if CONFIG["RESUME"]:
            ok_all = (not need_all) or (path_all and os.path.exists(path_all) and npy_ok(path_all))
            ok_l12 = (not need_last12) or (path_last12 and os.path.exists(path_last12) and npy_ok(path_last12))
            if ok_all and ok_l12:
                skipped += 1
                continue

        if not os.path.exists(wav_path):
            ext.failures.append({"uid": uid, "err": f"File not found: {wav_path}", "audio_path": wav_path})
            continue

        try:
            feats = ext.extract(wav_path)
            if feats is None:
                ext.failures.append({"uid": uid, "err": "Extracted None", "audio_path": wav_path})
                continue

            if need_all and ("avg_all" in feats) and path_all:
                if (not CONFIG["RESUME"]) or (not os.path.exists(path_all)) or (not npy_ok(path_all)):
                    ensure_dir(os.path.dirname(path_all))
                    np.save(path_all, feats["avg_all"])
                ext.metadata_all.append({"uid": uid, "shape": str(feats["avg_all"].shape), "path": path_all})

            if need_last12 and ("avg_last12" in feats) and path_last12:
                if (not CONFIG["RESUME"]) or (not os.path.exists(path_last12)) or (not npy_ok(path_last12)):
                    ensure_dir(os.path.dirname(path_last12))
                    np.save(path_last12, feats["avg_last12"])
                ext.metadata_last12.append({"uid": uid, "shape": str(feats["avg_last12"].shape), "path": path_last12})

            processed += 1

        except Exception as e:
            ext.failures.append({"uid": uid, "err": str(e), "audio_path": wav_path})
            logging.error(traceback.format_exc())

    print(f" WavLM Done. Processed={processed}, Skipped={skipped}, Failures={len(ext.failures)}")
    ext.save_metadata()


def run_whisper(ext: WhisperEncoderExtractor, df: pd.DataFrame):
    if "audio_path" not in df.columns:
        raise ValueError("Manifest missing required column: audio_path")

    print(" Start Task: Whisper encoder (no ASR)")
    skipped = 0
    processed = 0

    for r in tqdm(df.itertuples(index=False), total=len(df), desc="whisper_encoder"):
        uid = str(getattr(r, "uid"))
        wav_path = getattr(r, "audio_path")
        if not isinstance(wav_path, str) or not wav_path:
            wav_path = os.path.join(CONFIG["EMIT_AUDIO_ROOT"], f"{uid}.wav")

        save_path = os.path.join(ext.save_dir, f"{uid}.npy")

        if CONFIG["RESUME"] and os.path.exists(save_path) and npy_ok(save_path):
            skipped += 1
            continue

        if not os.path.exists(wav_path):
            ext.failures.append({"uid": uid, "err": f"File not found: {wav_path}", "audio_path": wav_path})
            continue

        try:
            feat = ext.extract(wav_path)
            if feat is None:
                ext.failures.append({"uid": uid, "err": "Extracted None", "audio_path": wav_path})
                continue
            ensure_dir(os.path.dirname(save_path))
            np.save(save_path, feat)
            ext.metadata.append({"uid": uid, "shape": str(feat.shape), "path": save_path})
            processed += 1
        except Exception as e:
            ext.failures.append({"uid": uid, "err": str(e), "audio_path": wav_path})
            logging.error(traceback.format_exc())

    print(f" Whisper Done. Processed={processed}, Skipped={skipped}, Failures={len(ext.failures)}")
    ext.save_metadata()


def main():
    logging.basicConfig(
        filename=CONFIG["LOG_PATH"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    ensure_dir(CONFIG["OUTPUT_ROOT"])
    df = read_manifest_to_df(CONFIG["MANIFEST_PATH"])
    print(f">>> Manifest loaded: {len(df)} rows")

    if "audio_path" not in df.columns:
        raise ValueError("Manifest missing required column: audio_path")

    if CONFIG["RUN_SEGMENT_WAV"]:
        run_segment_wavs(df)

    # 依次跑，避免同时占用显存
    if CONFIG["RUN_WAVLM_LAYERAVG"]:
        ext = None
        try:
            ext = WavLMLayerAvgExtractor(CONFIG["DEVICE"])
            run_wavlm_layeravg(ext, df)
        finally:
            cleanup_obj(ext)

    if CONFIG["RUN_WHISPER"]:
        ext = None
        try:
            ext = WhisperEncoderExtractor(CONFIG["DEVICE"])
            run_whisper(ext, df)
        finally:
            cleanup_obj(ext)

    print("All done.")


if __name__ == "__main__":
    main()
