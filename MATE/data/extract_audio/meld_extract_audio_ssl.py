#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MELD / MERBench-style audio extraction (aligned version)

What this script exports
------------------------
1) WavLM-Large avg_all24   -> [T, 1024]  frame-like sequence
2) WavLM-Large avg_last12  -> [T, 1024]  frame-like sequence
3) Whisper-Large-v3 encoder -> [T, 1280] frame-like sequence,
   cropped to VALID audio duration (not the full padded 30s window)

Design goal
-----------
Make the extraction details as close as possible to the MERBench-style
audio extraction logic you provided:

- mono
- resample to 16k
- cap length to 30s
- mean removal (DC offset removal)
- WavLM stored as frame-like [T,1024] sequences
- Whisper stored as frame-like [T,1280] sequences
- Whisper cropped to valid duration, not keeping the whole padded 30s window

Important
---------
To align with MERBench-style exports, this script saves FRAME sequences only
by default. It does NOT save utterance-level pooled vectors unless you
explicitly turn that on.

Edit CFG before running.
"""

from __future__ import annotations

import gc
import json
import os
import traceback
from pathlib import Path
from typing import Dict, Any, Iterator, Optional

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2FeatureExtractor, WhisperModel, WhisperFeatureExtractor

CFG = {
    # input
    "manifest_jsonl": "data/manifest/manifest.jsonl",

    # output root
    "output_root": "data/features",

    # device
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # audio preprocessing
    "target_sr": 16000,
    "max_sec": 30.0,                 # MERBench-style length cap
    "do_zmuv": False,                # keep False for compatibility with the original extraction setup

    # too-short audio fallback
    "min_samples_wavlm": 1600,       # compatible with the layer-averaged WavLM export
    "min_samples_whisper": 800,      # compatible with the Whisper export

    # models
    "wavlm_id": "microsoft/wavlm-large",
    "whisper_id": "openai/whisper-large-v3",

    # export switches
    "export_wavlm_avg_all24": True,
    "export_wavlm_avg_last12": True,
    "export_whisper_encoder": True,

    # exact-alignment choice:
    # Save frame-level sequences by default.
    "save_frame_only": True,

    # optional convenience export (OFF by default to keep alignment)
    "also_save_utt_mean": False,

    # resume
    "skip_if_exists_and_ok": True,

    # known bad samples
    "bad_uids": {
        "train_dia125_utt3",
        "val_dia110_utt7",
    },

    # logs
    "failure_jsonl": "_failures_audio_extract_aligned.jsonl",
}

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def npy_ok(path: Path) -> bool:
    try:
        arr = np.load(path, mmap_mode="r")
        return isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] >= 1
    except Exception:
        return False

def load_audio_1d(wav_path: str, target_sr: int, max_sec: float, do_zmuv: bool) -> Optional[torch.Tensor]:
    """
    MERBench-style preprocessing:
      read -> mono -> resample -> cap length -> mean removal -> optional ZMUV
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

        # cap length
        if max_sec and max_sec > 0:
            max_len = int(max_sec * target_sr)
            if wav.numel() > max_len:
                wav = wav[:max_len]

        # mean removal (DC offset removal)
        if wav.numel() > 0:
            wav = wav - wav.mean()

        # optional ZMUV
        if do_zmuv and wav.numel() > 1:
            std = wav.std()
            if std > 1e-7:
                wav = wav / std

        return wav.contiguous()
    except Exception:
        return None

def save_frame_and_optional_utt(base_dir: Path, uid: str, frame_arr: np.ndarray):
    ensure_dir(base_dir / "frame")
    np.save(base_dir / "frame" / f"{uid}.npy", frame_arr.astype(np.float32))

    if CFG["also_save_utt_mean"]:
        ensure_dir(base_dir / "utt")
        utt = frame_arr.mean(axis=0).astype(np.float32)
        np.save(base_dir / "utt" / f"{uid}.npy", utt)

def need_skip(base_dir: Path, uid: str) -> bool:
    frame_path = base_dir / "frame" / f"{uid}.npy"
    if not frame_path.exists():
        return False
    if not npy_ok(frame_path):
        return False

    if CFG["also_save_utt_mean"]:
        utt_path = base_dir / "utt" / f"{uid}.npy"
        if not utt_path.exists():
            return False
        try:
            utt = np.load(utt_path, mmap_mode="r")
            if not (isinstance(utt, np.ndarray) and utt.ndim == 1 and utt.shape[0] >= 1):
                return False
        except Exception:
            return False

    return True

class WavLMLayerAvgExtractor:
    def __init__(self, device: str):
        self.device = device
        self.target_sr = int(CFG["target_sr"])

        self.export_all = bool(CFG["export_wavlm_avg_all24"])
        self.export_last12 = bool(CFG["export_wavlm_avg_last12"])

        self.dir_all = Path(CFG["output_root"]) / "wavlm_large_avg_all24"
        self.dir_last12 = Path(CFG["output_root"]) / "wavlm_large_avg_last12"
        if self.export_all:
            ensure_dir(self.dir_all / "frame")
        if self.export_last12:
            ensure_dir(self.dir_last12 / "frame")
        if CFG["also_save_utt_mean"]:
            if self.export_all:
                ensure_dir(self.dir_all / "utt")
            if self.export_last12:
                ensure_dir(self.dir_last12 / "utt")

        dtype = torch.float16 if (device != "cpu" and torch.cuda.is_available()) else torch.float32
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(CFG["wavlm_id"])
        self.model = WavLMModel.from_pretrained(CFG["wavlm_id"], torch_dtype=dtype).to(device).eval()
        self.model.config.output_hidden_states = True

    @torch.no_grad()
    def extract(self, wav_path: str) -> Optional[Dict[str, np.ndarray]]:
        wav_1d = load_audio_1d(
            wav_path,
            target_sr=self.target_sr,
            max_sec=float(CFG["max_sec"]),
            do_zmuv=bool(CFG["do_zmuv"]),
        )
        if wav_1d is None:
            return None

        if wav_1d.numel() < int(CFG["min_samples_wavlm"]):
            out = {}
            if self.export_all:
                out["avg_all24"] = np.zeros((1, 1024), dtype=np.float32)
            if self.export_last12:
                out["avg_last12"] = np.zeros((1, 1024), dtype=np.float32)
            return out

        inputs = self.processor(wav_1d.cpu().numpy(), sampling_rate=self.target_sr, return_tensors="pt")
        x = inputs.input_values.to(self.device)

        if self.model.dtype == torch.float16:
            x = x.half()

        out = self.model(x)
        hs = out.hidden_states  # (emb, layer1, ..., layer24)

        if hs is None or len(hs) <= 1:
            feat = out.last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)
            return {"avg_all24": feat} if self.export_all else {"avg_last12": feat}

        layers = hs[1:]  # skip embedding
        results: Dict[str, np.ndarray] = {}

        if self.export_all:
            acc = None
            for h in layers:
                h32 = h.float()
                acc = h32 if acc is None else (acc + h32)
            mean_all = (acc / float(len(layers))).squeeze(0)  # [T,1024]
            results["avg_all24"] = mean_all.cpu().numpy().astype(np.float32)

        if self.export_last12:
            sel = layers[-12:] if len(layers) >= 12 else layers
            acc = None
            for h in sel:
                h32 = h.float()
                acc = h32 if acc is None else (acc + h32)
            mean_last12 = (acc / float(len(sel))).squeeze(0)  # [T,1024]
            results["avg_last12"] = mean_last12.cpu().numpy().astype(np.float32)

        return results

class WhisperEncoderExtractor:
    def __init__(self, device: str):
        self.device = device
        self.target_sr = int(CFG["target_sr"])

        self.dir_out = Path(CFG["output_root"]) / "whisper_large_v3_encoder"
        ensure_dir(self.dir_out / "frame")
        if CFG["also_save_utt_mean"]:
            ensure_dir(self.dir_out / "utt")

        dtype = torch.float16 if (device != "cpu" and torch.cuda.is_available()) else torch.float32
        self.processor = WhisperFeatureExtractor.from_pretrained(CFG["whisper_id"])
        self.model = WhisperModel.from_pretrained(CFG["whisper_id"], torch_dtype=dtype).to(device).eval()

    @torch.no_grad()
    def extract(self, wav_path: str) -> Optional[np.ndarray]:
        wav_1d = load_audio_1d(
            wav_path,
            target_sr=self.target_sr,
            max_sec=float(CFG["max_sec"]),
            do_zmuv=bool(CFG["do_zmuv"]),
        )
        if wav_1d is None:
            return None

        if wav_1d.numel() < int(CFG["min_samples_whisper"]):
            return np.zeros((1, 1280), dtype=np.float32)

        real_len_sec = float(wav_1d.numel()) / float(self.target_sr)

        inputs = self.processor(wav_1d.cpu().numpy(), sampling_rate=self.target_sr, return_tensors="pt")
        features = inputs.input_features.to(self.device)

        if self.model.dtype == torch.float16:
            features = features.half()

        out = self.model.encoder(features)
        hs = out.last_hidden_state.squeeze(0).cpu().numpy()  # [1500,1280] padded window output

        # Crop to valid duration, not full padded 30s window
        # Whisper encoder time resolution is approximately 20ms/frame in this export logic.
        valid_frames = int(real_len_sec / 0.02)
        valid_frames = min(valid_frames + 2, hs.shape[0])  # small tolerance, same style as your uploaded code
        valid_frames = max(1, valid_frames)

        return hs[:valid_frames].astype(np.float32)

def cleanup(*objs):
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    out_root = Path(CFG["output_root"])
    ensure_dir(out_root)

    records = list(iter_jsonl(CFG["manifest_jsonl"]))
    print(f"Loaded {len(records)} utterances")
    print(f"Device: {CFG['device']}")
    print("Alignment mode: FRAME sequences (MERBench-style)")

    failures = []

    wavlm_ext = None
    whisper_ext = None

    try:
        if CFG["export_wavlm_avg_all24"] or CFG["export_wavlm_avg_last12"]:
            print("Loading WavLM-Large...")
            wavlm_ext = WavLMLayerAvgExtractor(CFG["device"])

        if CFG["export_whisper_encoder"]:
            print("Loading Whisper-Large-v3 encoder...")
            whisper_ext = WhisperEncoderExtractor(CFG["device"])

        for rec in tqdm(records, desc="extract_audio_aligned"):
            uid = str(rec.get("uid") or rec.get("utt_id") or "").strip()
            wav_path = str(rec.get("audio_path") or "").strip()

            if not uid:
                continue
            if uid in CFG["bad_uids"]:
                failures.append({"uid": uid, "error": "skip_bad_uid"})
                continue
            if not wav_path or not os.path.exists(wav_path):
                failures.append({"uid": uid, "error": "missing_audio_path", "audio_path": wav_path})
                continue

            # WavLM
            if wavlm_ext is not None:
                try:
                    if CFG["export_wavlm_avg_all24"]:
                        if not (CFG["skip_if_exists_and_ok"] and need_skip(wavlm_ext.dir_all, uid)):
                            feats = wavlm_ext.extract(wav_path)
                            if feats is None:
                                failures.append({"uid": uid, "error": "wavlm_extract_none", "audio_path": wav_path})
                            else:
                                if "avg_all24" in feats:
                                    save_frame_and_optional_utt(wavlm_ext.dir_all, uid, feats["avg_all24"])
                                if CFG["export_wavlm_avg_last12"]:
                                    if not (CFG["skip_if_exists_and_ok"] and need_skip(wavlm_ext.dir_last12, uid)):
                                        if "avg_last12" in feats:
                                            save_frame_and_optional_utt(wavlm_ext.dir_last12, uid, feats["avg_last12"])
                        else:
                            if CFG["export_wavlm_avg_last12"]:
                                if not (CFG["skip_if_exists_and_ok"] and need_skip(wavlm_ext.dir_last12, uid)):
                                    feats = wavlm_ext.extract(wav_path)
                                    if feats is None:
                                        failures.append({"uid": uid, "error": "wavlm_extract_none", "audio_path": wav_path})
                                    else:
                                        if "avg_last12" in feats:
                                            save_frame_and_optional_utt(wavlm_ext.dir_last12, uid, feats["avg_last12"])
                    else:
                        if CFG["export_wavlm_avg_last12"] and not (CFG["skip_if_exists_and_ok"] and need_skip(wavlm_ext.dir_last12, uid)):
                            feats = wavlm_ext.extract(wav_path)
                            if feats is None:
                                failures.append({"uid": uid, "error": "wavlm_extract_none", "audio_path": wav_path})
                            else:
                                if "avg_last12" in feats:
                                    save_frame_and_optional_utt(wavlm_ext.dir_last12, uid, feats["avg_last12"])
                except Exception as e:
                    failures.append({"uid": uid, "error": f"wavlm_failed:{repr(e)}", "audio_path": wav_path})

            # Whisper
            if whisper_ext is not None:
                try:
                    if not (CFG["skip_if_exists_and_ok"] and need_skip(whisper_ext.dir_out, uid)):
                        feat = whisper_ext.extract(wav_path)
                        if feat is None:
                            failures.append({"uid": uid, "error": "whisper_extract_none", "audio_path": wav_path})
                        else:
                            save_frame_and_optional_utt(whisper_ext.dir_out, uid, feat)
                except Exception as e:
                    failures.append({"uid": uid, "error": f"whisper_failed:{repr(e)}", "audio_path": wav_path})

    finally:
        cleanup(wavlm_ext, whisper_ext)

    if failures:
        fail_path = out_root / CFG["failure_jsonl"]
        with open(fail_path, "w", encoding="utf-8") as f:
            for row in failures:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[WARN] failures saved to {fail_path}")

    print("Done.")

if __name__ == "__main__":
    main()
