#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CREMA-D / IEMOCAP 通用：WavLM 离线特征提取（层平均版：avg_all + avg_last12）
- 只需改 CONFIG 里的路径，直接运行
- 每条音频只 forward 一次 WavLM，同时导出两套特征
- 输出 shape: [T, 1024] float32，能直接喂给CNN/Transformer 头

CSV 要求：
  - 必须有列：audio_path
  - 推荐有列：uid（用于对齐 utt_id）；没有则用文件名去掉后缀
"""

import os
import gc
import json
import ntpath
import logging
import traceback
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio
from transformers import WavLMModel, Wav2Vec2FeatureExtractor


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


CONFIG = {
    # 你的 CSV 路径（需包含 audio_path；最好还有 uid）
    "CSV_PATH": r"manifest_cremad_vote_av.csv",

    # 输出根目录（会在下面自动创建两个子目录）
    "OUTPUT_ROOT": r"audio_reextract",

    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # 音频预处理
    "TARGET_SR": 16000,
    "MAX_SEC": 30.0,          # >0 则截断到 MAX_SEC 秒；0 则不截断
    "MIN_SAMPLES": 1600,      # 太短的音频直接输出全 0

    # WavLM 模型
    "WAVLM_ID": "wavlm-large",

    # 导出两种层平均（固定输出这两套，符合你“一个all平均+一个last12平均”的需求）
    "EXPORT_AVG_ALL": True,
    "EXPORT_AVG_LAST12": True,
    "LAST_N": 12,

    # 若已存在 npy 是否跳过
    "SKIP_IF_EXISTS_AND_OK": True,

    # 日志
    "LOG_PATH": "wavlm_layeravg_extract.log",
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


def load_audio_1d(wav_path: str, target_sr: int) -> Optional[torch.Tensor]:
    """
    统一预处理：读 -> 单声道 -> 重采样 -> 截断 -> 去DC
    返回：1D tensor (num_samples,)
    """
    try:
        wav, sr = torchaudio.load(wav_path)  # [C, N]
        if wav.numel() == 0:
            return None

        # mono
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        wav = wav.squeeze(0)  # [N]

        # resample
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            wav = resampler(wav)

        # cap length
        max_sec = float(CONFIG["MAX_SEC"])
        if max_sec > 0:
            max_len = int(max_sec * target_sr)
            if wav.numel() > max_len:
                wav = wav[:max_len]

        # DC offset removal
        if wav.numel() > 0:
            wav = wav - wav.mean()

        return wav
    except Exception:
        logging.error(traceback.format_exc())
        return None


def file_id_from_row(row: Any) -> str:
    if hasattr(row, "uid"):
        return str(getattr(row, "uid"))
    # fallback: filename stem
    base = ntpath.basename(str(getattr(row, "audio_path")))
    stem = os.path.splitext(base)[0]
    return stem


class WavLMLayerAvgExtractor:
    """
    一次 forward，导出两套：
      - avg_all: 所有 Transformer 层平均（跳过 embedding）
      - avg_last12: 最后12层平均（跳过 embedding）
    """

    def __init__(self, device: str):
        self.device = device
        self.target_sr = int(CONFIG["TARGET_SR"])

        self.export_all = bool(CONFIG["EXPORT_AVG_ALL"])
        self.export_last12 = bool(CONFIG["EXPORT_AVG_LAST12"])
        self.last_n = int(CONFIG["LAST_N"])

        assert self.export_all or self.export_last12, "至少导出一种特征。"

        # 输出目录
        self.dir_all = os.path.join(CONFIG["OUTPUT_ROOT"], "audio_wavlm_avg_all")
        self.dir_last12 = os.path.join(CONFIG["OUTPUT_ROOT"], "audio_wavlm_avg_last12")
        if self.export_all:
            ensure_dir(self.dir_all)
        if self.export_last12:
            ensure_dir(self.dir_last12)

        self.metadata_all: List[dict] = []
        self.metadata_last12: List[dict] = []
        self.failures: List[dict] = []

        print("Loading WavLM (with hidden_states)...")
        model_id = CONFIG["WAVLM_ID"]
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

        dtype = torch.float16 if (device != "cpu" and torch.cuda.is_available()) else torch.float32
        self.model = WavLMModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        # 关键：打开 hidden_states
        self.model.config.output_hidden_states = True
        print(f"WavLM Loaded. dtype={self.model.dtype} device={device}")

    @torch.no_grad()
    def extract(self, wav_path: str) -> Optional[Dict[str, np.ndarray]]:
        wav_1d = load_audio_1d(wav_path, self.target_sr)
        if wav_1d is None:
            return None
        if wav_1d.numel() < int(CONFIG["MIN_SAMPLES"]):
            out = {}
            if self.export_all:
                out["avg_all"] = np.zeros((1, 1024), dtype=np.float32)
            if self.export_last12:
                out["avg_last12"] = np.zeros((1, 1024), dtype=np.float32)
            return out

        inputs = self.processor(wav_1d.cpu().numpy(), sampling_rate=self.target_sr, return_tensors="pt")
        x = inputs.input_values.to(self.device)

        if self.model.dtype == torch.float16:
            x = x.half()

        out = self.model(x)
        hs = out.hidden_states  # tuple: (emb, layer1, ..., layerL)

        if hs is None or len(hs) <= 1:
            # 极端兜底：退回 last_hidden_state
            feat = out.last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)
            return {"avg_all": feat} if self.export_all else {"avg_last12": feat}

        # 跳过 embedding 层，只用 transformer 层
        layers = hs[1:]  # len = num_layers
        # 每层 shape: [B, T, D]

        results: Dict[str, np.ndarray] = {}

        # --- avg_all ---
        if self.export_all:
            acc = None
            for h in layers:
                h32 = h.float()
                acc = h32 if acc is None else (acc + h32)
            mean_all = (acc / float(len(layers))).squeeze(0)  # [T, D]
            results["avg_all"] = mean_all.cpu().numpy().astype(np.float32)

        # --- avg_last12 ---
        if self.export_last12:
            sel = layers[-self.last_n:] if len(layers) >= self.last_n else layers
            acc = None
            for h in sel:
                h32 = h.float()
                acc = h32 if acc is None else (acc + h32)
            mean_last = (acc / float(len(sel))).squeeze(0)  # [T, D]
            results["avg_last12"] = mean_last.cpu().numpy().astype(np.float32)

        return results

    def save_metadata(self):
        if self.export_all:
            df = pd.DataFrame(self.metadata_all)
            p = os.path.join(self.dir_all, "_metadata.csv")
            df.to_csv(p, index=False)
            print(f" [avg_all] metadata -> {p}")

        if self.export_last12:
            df = pd.DataFrame(self.metadata_last12)
            p = os.path.join(self.dir_last12, "_metadata.csv")
            df.to_csv(p, index=False)
            print(f"[avg_last12] metadata -> {p}")

        if self.failures:
            p = os.path.join(CONFIG["OUTPUT_ROOT"], "_failures.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for r in self.failures:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"failures {len(self.failures)} -> {p}")


def cleanup(obj):
    try:
        del obj
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    logging.basicConfig(
        filename=CONFIG["LOG_PATH"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    csv_path = CONFIG["CSV_PATH"]
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    ensure_dir(CONFIG["OUTPUT_ROOT"])
    df = pd.read_csv(csv_path)
    if "audio_path" not in df.columns:
        raise ValueError("CSV missing required column: audio_path")

    ext = None
    try:
        ext = WavLMLayerAvgExtractor(CONFIG["DEVICE"])
        print(" Start WavLM layer-avg extraction...")

        skipped = 0
        processed = 0

        for r in tqdm(df.itertuples(index=False), total=len(df), desc="wavlm_layeravg"):
            wav_path = str(getattr(r, "audio_path"))
            fid = file_id_from_row(r)

            try:
                feats = ext.extract(wav_path)
                if feats is None:
                    skipped += 1
                    continue

                # 保存 avg_all
                if "avg_all" in feats and ext.export_all:
                    save_path = os.path.join(ext.dir_all, f"{fid}.npy")
                    if CONFIG["SKIP_IF_EXISTS_AND_OK"] and os.path.exists(save_path) and npy_ok(save_path):
                        skipped += 1
                    else:
                        np.save(save_path, feats["avg_all"])
                    ext.metadata_all.append({"file_id": fid, "shape": str(feats["avg_all"].shape), "path": save_path})

                # 保存 avg_last12
                if "avg_last12" in feats and ext.export_last12:
                    save_path = os.path.join(ext.dir_last12, f"{fid}.npy")
                    if CONFIG["SKIP_IF_EXISTS_AND_OK"] and os.path.exists(save_path) and npy_ok(save_path):
                        skipped += 1
                    else:
                        np.save(save_path, feats["avg_last12"])
                    ext.metadata_last12.append({"file_id": fid, "shape": str(feats["avg_last12"].shape), "path": save_path})

                processed += 1

            except Exception as e:
                ext.failures.append({"id": fid, "audio_path": wav_path, "err": str(e)})
                logging.error(traceback.format_exc())

        print(f"Done. Processed: {processed}, Skipped: {skipped}, Failures: {len(ext.failures)}")
        ext.save_metadata()

    finally:
        cleanup(ext)


if __name__ == "__main__":
    main()
