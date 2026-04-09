"""
【文件作用】模块说明：请阅读本文件顶部注释。

（如果你在 IDE 里阅读代码：先看本段，再往下看实现。）
"""


# cremad_audio_extract_lite.py
# 功能：CREMA-D 音频特征提取（精简版：仅 WavLM + Whisper）
# 剔除了 Emotion2Vec，无需安装 funasr

import os
import gc
import json
import ntpath
import logging
import traceback
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio

from transformers import (
    WavLMModel, Wav2Vec2FeatureExtractor,
    WhisperModel, WhisperFeatureExtractor
)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


CONFIG = {
    # 你的 CSV 路径
    "CSV_PATH": r"manifest_cremad_vote_av.csv",

    # 输出目录
    "OUTPUT_ROOT": r"audio",

    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # 开关
    "RUN_WAVLM": True,
    "RUN_WHISPER": True,

    # 音频预处理参数
    "MAX_SEC": 30.0,  # 超过30秒截断
    "RESUME": True,  # 断点续跑
    "DO_ZMUV": False,  # 是否做 Z-Score 标准化 (SSL模型通常False)
    "MIN_SAMPLES": 800,  # 最小采样点数 (50ms)

    "MODELS": {
        "wavlm": {
            "id": "wavlm-large",
            "sr": 16000
        },
        "whisper": {
            "id": "whisper-large-v3",
            "sr": 16000
        }
    }
}
# ============================================================

logging.basicConfig(
    filename="cremad_extract_lite.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def npy_ok(p: str) -> bool:
    try:
        arr = np.load(p, mmap_mode="r")
        if arr.ndim != 2: return False
        if arr.shape[0] < 1 or arr.shape[1] < 1: return False
        return True
    except Exception:
        return False



class BaseExtractor:
    def __init__(self, name: str, sr: int, device: str):
        self.name = name
        self.target_sr = sr
        self.device = device
        self.save_dir = os.path.join(CONFIG["OUTPUT_ROOT"], name)
        ensure_dir(self.save_dir)
        print(f"[{name}] 初始化中... (保存路径: {self.save_dir})")
        self.metadata = []
        self.failures = []

    def load_audio_1d(self, wav_path: str) -> Optional[torch.Tensor]:
        """
        统一预处理：读 -> 单声道 -> 重采样 -> 截断 -> 去DC -> (可选ZMUV)
        """
        try:
            wav, sr = torchaudio.load(wav_path)  # [C, T]
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0)  # [T]

            # Resample
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                wav = resampler(wav)

            # Length Cap
            max_sec = float(CONFIG["MAX_SEC"])
            if max_sec > 0:
                max_len = int(max_sec * self.target_sr)
                if wav.numel() > max_len:
                    wav = wav[:max_len]

            if wav.numel() > 0:
                # 去除直流偏移 (DC offset)
                wav = wav - wav.mean()
                # 可选 Z-Score
                if CONFIG["DO_ZMUV"]:
                    std = wav.std()
                    if std > 1e-7:
                        wav = wav / std

            return wav.contiguous()
        except Exception as e:
            logging.error(f"Error loading {wav_path}: {e}")
            return None

    def save_metadata(self):
        if not self.metadata:
            return
        df = pd.DataFrame(self.metadata)
        csv_path = os.path.join(self.save_dir, "metadata.csv")
        df.to_csv(csv_path, index=False)
        print(f" [{self.name}] Metadata 已保存至 {csv_path}")

        if self.failures:
            fail_path = os.path.join(self.save_dir, "_failures.jsonl")
            with open(fail_path, "w", encoding="utf-8") as f:
                for r in self.failures:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{self.name}] 失败 {len(self.failures)} 条 -> {fail_path}")

    def process(self, wav_path: str) -> Optional[np.ndarray]:
        raise NotImplementedError



class WavLMExtractor(BaseExtractor):
    def __init__(self, device: str):
        super().__init__("audio_wavlm", CONFIG["MODELS"]["wavlm"]["sr"], device)
        print("⏳ Loading WavLM Large...")
        model_id = CONFIG["MODELS"]["wavlm"]["id"]
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

        # 显存优化：GPU 使用 fp16
        dtype = torch.float16 if (device != "cpu" and torch.cuda.is_available()) else torch.float32
        self.model = WavLMModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        print(" WavLM Loaded.")

    @torch.no_grad()
    def process(self, wav_path: str) -> Optional[np.ndarray]:
        wav_1d = self.load_audio_1d(wav_path)
        if wav_1d is None: return None
        if wav_1d.numel() < CONFIG["MIN_SAMPLES"]:
            return np.zeros((1, 1024), dtype=np.float32)

        # process inputs
        inputs = self.processor(wav_1d.cpu().numpy(), sampling_rate=self.target_sr, return_tensors="pt")
        x = inputs.input_values.to(self.device)

        # dtype match
        if self.model.dtype == torch.float16:
            x = x.half()

        out = self.model(x)
        feat = out.last_hidden_state.squeeze(0).cpu().numpy()  # [T, 1024]
        return feat.astype(np.float32)



class WhisperExtractor(BaseExtractor):
    def __init__(self, device: str):
        super().__init__("audio_whisper", CONFIG["MODELS"]["whisper"]["sr"], device)
        print("Loading Whisper Large-v3...")
        model_id = CONFIG["MODELS"]["whisper"]["id"]
        self.processor = WhisperFeatureExtractor.from_pretrained(model_id)

        dtype = torch.float16 if (device != "cpu" and torch.cuda.is_available()) else torch.float32
        self.model = WhisperModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        print(" Whisper Loaded.")

    @torch.no_grad()
    def process(self, wav_path: str) -> Optional[np.ndarray]:
        wav_1d = self.load_audio_1d(wav_path)
        if wav_1d is None: return None
        if wav_1d.numel() < CONFIG["MIN_SAMPLES"]:
            return np.zeros((1, 1280), dtype=np.float32)

        real_len_sec = wav_1d.numel() / self.target_sr

        # WhisperProcessor pads to 30s internally
        inputs = self.processor(wav_1d.cpu().numpy(), sampling_rate=self.target_sr, return_tensors="pt")
        features = inputs.input_features.to(self.device)

        if self.model.dtype == torch.float16:
            features = features.half()

        out = self.model.encoder(features)
        hs = out.last_hidden_state.squeeze(0).cpu().numpy()  # [1500, 1280]

        # 帧数截断优化: 20ms/frame
        valid_frames = int(real_len_sec / 0.02)
        valid_frames = min(valid_frames + 2, hs.shape[0])
        valid_frames = max(1, valid_frames)

        return hs[:valid_frames].astype(np.float32)



def cleanup_extractor(extractor):
    if extractor is not None:
        name = extractor.name
        print(f"Unloading {name}...")
        try:
            del extractor.model
            del extractor.processor
        except AttributeError:
            pass
        del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(" Memory cleared.\n")



def run_one_task(ext: BaseExtractor, df: pd.DataFrame):
    if "audio_path" not in df.columns:
        raise ValueError("CSV missing 'audio_path' column.")
    has_uid = "uid" in df.columns

    print(f" Start Task: {ext.name}")

    skipped = 0
    processed = 0

    for r in tqdm(df.itertuples(index=False), total=len(df), desc=ext.name):
        wav_path = str(getattr(r, "audio_path"))

        if has_uid:
            file_id = str(getattr(r, "uid"))
        else:
            file_id = os.path.splitext(ntpath.basename(wav_path))[0]

        save_path = os.path.join(ext.save_dir, f"{file_id}.npy")

        # Resume
        if CONFIG["RESUME"] and os.path.exists(save_path):
            if npy_ok(save_path):
                skipped += 1
                continue

        # File Check
        if not os.path.exists(wav_path):
            ext.failures.append({"id": file_id, "err": f"File not found: {wav_path}"})
            continue

        try:
            feat = ext.process(wav_path)
            if feat is None:
                ext.failures.append({"id": file_id, "err": "Extracted None"})
                continue

            np.save(save_path, feat)
            ext.metadata.append({"file_id": file_id, "shape": str(feat.shape), "path": save_path})
            processed += 1

        except Exception as e:
            ext.failures.append({"id": file_id, "err": str(e)})
            logging.error(traceback.format_exc())

    print(f"🏁 {ext.name} Done. Processed: {processed}, Skipped: {skipped}, Failures: {len(ext.failures)}")
    ext.save_metadata()


def main():
    if not os.path.exists(CONFIG["CSV_PATH"]):
        print(f" CSV not found: {CONFIG['CSV_PATH']}")
        return

    print(f">>> Reading CSV: {CONFIG['CSV_PATH']}")
    df = pd.read_csv(CONFIG['CSV_PATH'])
    print(f">>> Total Samples: {len(df)}")

    # 任务列表：仅保留 WavLM 和 Whisper
    tasks = [
        (CONFIG["RUN_WAVLM"], WavLMExtractor),
        (CONFIG["RUN_WHISPER"], WhisperExtractor),
    ]

    ensure_dir(CONFIG["OUTPUT_ROOT"])

    for should_run, cls in tasks:
        if not should_run: continue

        ext = None
        try:
            ext = cls(CONFIG["DEVICE"])
            run_one_task(ext, df)
        except Exception as e:
            print(f" Critical Error in {cls.__name__}: {e}")
            logging.error(traceback.format_exc())
        finally:
            cleanup_extractor(ext)


if __name__ == "__main__":
    main()
