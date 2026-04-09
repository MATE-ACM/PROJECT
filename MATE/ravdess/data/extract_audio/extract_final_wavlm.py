#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import gc
import json
import ntpath
import logging
import traceback
from typing import Optional, Any, List

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


CONFIG = {
    "CSV_PATH": r"meta_all_features.csv",
    "OUTPUT_ROOT": r"feature",

    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # 开关
    "RUN_WAVLM_LASTLAYER": True,
    "RUN_WHISPER": False,

    # 音频预处理
    "TARGET_SR": 16000,
    "MAX_SEC": 30.0,          # >0 则截断到 MAX_SEC 秒；0 则不截断
    "MIN_SAMPLES": 800,       # 太短的音频直接输出全 0

    # 可选：ZMUV（SSL 一般不需要）
    "DO_ZMUV": False,

    # 模型
    "WAVLM_ID": "wavlm-large",
    "WHISPER_ID": "whisper-large-v3",


    "WAVLM_DIRNAME": "audio_wavlm-FINAL",

    # 若已存在 npy 是否跳过
    "RESUME": True,

    # 日志
    "LOG_PATH": "audio_extract_wavlm_lastlayer.log",
}
# ============================================================


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def npy_ok(p: str) -> bool:
    try:
        arr = np.load(p, mmap_mode="r")
        return (arr.ndim == 2) and (arr.shape[0] >= 1) and (arr.shape[1] >= 1)
    except Exception:
        return False


def file_id_from_row(row: Any) -> str:
    """优先 uid，否则从 audio_path 文件名推 id"""
    if hasattr(row, "uid"):
        return str(getattr(row, "uid"))
    base = ntpath.basename(str(getattr(row, "audio_path")))
    stem = os.path.splitext(base)[0]
    return stem


def load_audio_1d(wav_path: str, target_sr: int) -> Optional[torch.Tensor]:
    """
    统一预处理：读 -> 单声道 -> 重采样 -> 截断 -> 去DC -> (可选) ZMUV
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




class WavLMLastLayerExtractor:
    """导出 out.last_hidden_state (最后一层) 帧级特征：shape=[T, 1024]"""

    def __init__(self, device: str):
        self.name = "audio_wavlm"
        self.device = device
        self.target_sr = int(CONFIG["TARGET_SR"])

        self.save_dir = os.path.join(CONFIG["OUTPUT_ROOT"], str(CONFIG["WAVLM_DIRNAME"]))
        ensure_dir(self.save_dir)

        self.metadata: List[dict] = []
        self.failures: List[dict] = []

        print(" Loading WavLM (last_hidden_state)...")
        model_id = CONFIG["WAVLM_ID"]
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

        dtype = torch.float16 if (device != "cpu" and torch.cuda.is_available()) else torch.float32
        self.model = WavLMModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        print(f" WavLM Loaded. dtype={self.model.dtype} device={device}")

    @torch.no_grad()
    def extract(self, wav_path: str) -> Optional[np.ndarray]:
        wav_1d = load_audio_1d(wav_path, self.target_sr)
        if wav_1d is None:
            return None

        # too short
        if wav_1d.numel() < int(CONFIG["MIN_SAMPLES"]):
            return np.zeros((1, 1024), dtype=np.float32)

        inputs = self.processor(wav_1d.cpu().numpy(), sampling_rate=self.target_sr, return_tensors="pt")
        x = inputs.input_values.to(self.device)

        # dtype match
        if self.model.dtype == torch.float16:
            x = x.half()

        out = self.model(x)
        feat = out.last_hidden_state.squeeze(0).detach().cpu().numpy()  # [T,1024]
        return feat.astype(np.float32)

    def save_metadata(self):
        if self.metadata:
            p = os.path.join(self.save_dir, "_metadata.csv")
            pd.DataFrame(self.metadata).to_csv(p, index=False)
            print(f" [wavlm last] metadata -> {p}")
        if self.failures:
            p = os.path.join(CONFIG["OUTPUT_ROOT"], "_failures_wavlm.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for r in self.failures:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f" [wavlm] failures {len(self.failures)} -> {p}")




class WhisperExtractor:
    def __init__(self, device: str):
        self.name = "audio_whisper"
        self.device = device
        self.target_sr = int(CONFIG["TARGET_SR"])
        self.save_dir = os.path.join(CONFIG["OUTPUT_ROOT"], self.name)
        ensure_dir(self.save_dir)

        self.metadata: List[dict] = []
        self.failures: List[dict] = []

        print(" Loading Whisper...")
        model_id = CONFIG["WHISPER_ID"]
        self.processor = WhisperFeatureExtractor.from_pretrained(model_id)
        dtype = torch.float16 if (device != "cpu" and torch.cuda.is_available()) else torch.float32
        self.model = WhisperModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        print(f" Whisper Loaded. dtype={self.model.dtype} device={device}")

    @torch.no_grad()
    def extract(self, wav_path: str) -> Optional[np.ndarray]:
        wav_1d = load_audio_1d(wav_path, self.target_sr)
        if wav_1d is None:
            return None

        if wav_1d.numel() < int(CONFIG["MIN_SAMPLES"]):
            return np.zeros((1, 1280), dtype=np.float32)

        real_len_sec = wav_1d.numel() / float(self.target_sr)

        inputs = self.processor(wav_1d.cpu().numpy(), sampling_rate=self.target_sr, return_tensors="pt")
        feats = inputs.input_features.to(self.device)
        if self.model.dtype == torch.float16:
            feats = feats.half()

        out = self.model.encoder(feats)
        hs = out.last_hidden_state.squeeze(0).cpu().numpy()  # [T, 1280]

        # roughly 20ms/frame
        valid_frames = int(real_len_sec / 0.02)
        valid_frames = min(valid_frames + 2, hs.shape[0])
        valid_frames = max(1, valid_frames)

        return hs[:valid_frames].astype(np.float32)

    def save_metadata(self):
        if self.metadata:
            p = os.path.join(self.save_dir, "metadata.csv")
            pd.DataFrame(self.metadata).to_csv(p, index=False)
            print(f" [whisper] metadata -> {p}")
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


def run_wavlm_lastlayer(ext: WavLMLastLayerExtractor, df: pd.DataFrame):
    if "audio_path" not in df.columns:
        raise ValueError("Manifest missing required column: audio_path")
    has_uid = "uid" in df.columns

    print(" Start Task: WavLM last_hidden_state")
    skipped = 0
    processed = 0

    for r in tqdm(df.itertuples(index=False), total=len(df), desc="wavlm_last"):
        wav_path = str(getattr(r, "audio_path"))
        fid = str(getattr(r, "uid")) if has_uid else file_id_from_row(r)
        save_path = os.path.join(ext.save_dir, f"{fid}.npy")

        if CONFIG["RESUME"] and os.path.exists(save_path) and npy_ok(save_path):
            skipped += 1
            continue

        if not os.path.exists(wav_path):
            ext.failures.append({"id": fid, "err": f"File not found: {wav_path}", "audio_path": wav_path})
            continue

        try:
            feat = ext.extract(wav_path)
            if feat is None:
                ext.failures.append({"id": fid, "err": "Extracted None", "audio_path": wav_path})
                continue

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, feat)
            ext.metadata.append({"file_id": fid, "shape": str(feat.shape), "path": save_path})
            processed += 1

        except Exception as e:
            ext.failures.append({"id": fid, "err": str(e), "audio_path": wav_path})
            logging.error(traceback.format_exc())

    print(f" WavLM Done. Processed={processed}, Skipped={skipped}, Failures={len(ext.failures)}")
    ext.save_metadata()


def run_whisper(ext: WhisperExtractor, df: pd.DataFrame):
    if "audio_path" not in df.columns:
        raise ValueError("Manifest missing required column: audio_path")
    has_uid = "uid" in df.columns

    print(" Start Task: Whisper encoder")
    skipped = 0
    processed = 0

    for r in tqdm(df.itertuples(index=False), total=len(df), desc="whisper"):
        wav_path = str(getattr(r, "audio_path"))
        fid = str(getattr(r, "uid")) if has_uid else file_id_from_row(r)

        save_path = os.path.join(ext.save_dir, f"{fid}.npy")

        if CONFIG["RESUME"] and os.path.exists(save_path) and npy_ok(save_path):
            skipped += 1
            continue

        if not os.path.exists(wav_path):
            ext.failures.append({"id": fid, "err": f"File not found: {wav_path}", "audio_path": wav_path})
            continue

        try:
            feat = ext.extract(wav_path)
            if feat is None:
                ext.failures.append({"id": fid, "err": "Extracted None", "audio_path": wav_path})
                continue

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, feat)
            ext.metadata.append({"file_id": fid, "shape": str(feat.shape), "path": save_path})
            processed += 1

        except Exception as e:
            ext.failures.append({"id": fid, "err": str(e), "audio_path": wav_path})
            logging.error(traceback.format_exc())

    print(f" Whisper Done. Processed={processed}, Skipped={skipped}, Failures={len(ext.failures)}")
    ext.save_metadata()


def main():
    logging.basicConfig(
        filename=CONFIG["LOG_PATH"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    file_path = CONFIG["CSV_PATH"]
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return

    ensure_dir(CONFIG["OUTPUT_ROOT"])
    print(f">>> Reading Manifest: {file_path}")

    # 兼容 jsonl / csv
    if file_path.lower().endswith(".jsonl"):
        df = pd.read_json(file_path, lines=True)
    else:
        df = pd.read_csv(file_path)

    # 自动对齐 ID 列名：把 utt_id 改名为 uid，方便后续逻辑统一使用
    if "utt_id" in df.columns and "uid" not in df.columns:
        print("ℹ  Detected 'utt_id', renaming to 'uid' for processing...")
        df = df.rename(columns={"utt_id": "uid"})

    print(f">>> Total Samples: {len(df)}")

    if "audio_path" not in df.columns:
        raise ValueError("Manifest missing required column: audio_path")

    # 依次跑，避免同时占用显存
    if CONFIG["RUN_WAVLM_LASTLAYER"]:
        ext = None
        try:
            ext = WavLMLastLayerExtractor(CONFIG["DEVICE"])
            run_wavlm_lastlayer(ext, df)
        finally:
            cleanup_obj(ext)

    if CONFIG["RUN_WHISPER"]:
        ext = None
        try:
            ext = WhisperExtractor(CONFIG["DEVICE"])
            run_whisper(ext, df)
        finally:
            cleanup_obj(ext)

    print(" All done.")


if __name__ == "__main__":
    main()