#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MELD WavLM extractor (frame only)

This version follows the current extraction style and directory layout,
but changes the extractor to:
  - read manifest.jsonl
  - use uid / utt_id + audio_path from manifest
  - save ONLY frame-level .npy features
  - split long audio into 10-second chunks (MERBench-style)
  - keep your preferred layer averaging variants:
        1) avg_last12
        2) avg_all24

Compared with your current extractor:
  - REMOVE 30s truncation by default
  - REMOVE mean-removal by default
  - ADD 10s chunking before WavLM inference
  - KEEP frame output only

Output layout:
  <output_root>/wavlm_large_chunk10_avg_last12/frame/<uid>.npy
  <output_root>/wavlm_large_chunk10_avg_all24/frame/<uid>.npy

You should point feat_root to one of the frame dirs above when training.
"""
from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

CFG = {
    # input manifest produced by your meld_prepare_merbench_style.py
    "manifest_jsonl": "data/manifest/manifest.jsonl",

    # root dir to save extracted features
    "output_root": "data/features",

    # model/device
    "wavlm_id": "microsoft/wavlm-large",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_fp16": True,

    # audio preprocess
    "target_sr": 16000,
    "do_zmuv": False,
    "remove_dc": False,
    "max_sec": None,          # None = do NOT truncate; keeps closer to MERBench chunking behavior
    "chunk_sec": 10.0,        # MERBench-style long-audio chunking
    "min_samples": 1600,

    # exports
    "export_avg_last12": True,
    "export_avg_all24": True,

    # resume / skips
    "skip_if_exists_and_ok": True,
    "bad_uids": {
        "train_dia125_utt3",
        "val_dia110_utt7",
    },
    "failure_jsonl": "_failures_wavlm_chunk10_layeravg.jsonl",
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

def need_skip(base_dir: Path, uid: str) -> bool:
    frame_path = base_dir / "frame" / f"{uid}.npy"
    return frame_path.exists() and npy_ok(frame_path)

def load_audio_1d(
    wav_path: str,
    target_sr: int,
    do_zmuv: bool,
    remove_dc: bool,
    max_sec: Optional[float],
) -> Optional[torch.Tensor]:
    try:
        wav, sr = torchaudio.load(wav_path)  # [C, T]
        if wav.numel() == 0:
            return None

        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)

        # resample
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)

        # optional truncate
        if max_sec is not None and max_sec > 0:
            max_len = int(max_sec * target_sr)
            if wav.numel() > max_len:
                wav = wav[:max_len]

        # optional mean removal
        if remove_dc and wav.numel() > 0:
            wav = wav - wav.mean()

        # optional zmuv
        if do_zmuv and wav.numel() > 1:
            std = wav.std()
            if std > 1e-7:
                wav = wav / std

        return wav.contiguous()
    except Exception:
        return None

def split_into_batch(input_values: torch.Tensor, maxlen: int) -> torch.Tensor:
    """
    input_values: [1, wavlen]
    return:       [B, maxlen]
    MERBench-style padding/chunking.
    """
    if input_values.ndim != 2 or input_values.shape[0] != 1:
        raise ValueError(f"expected [1, T], got {tuple(input_values.shape)}")

    wavlen = int(input_values.shape[1])
    if wavlen <= maxlen:
        return input_values

    tgtlen = math.ceil(wavlen / maxlen) * maxlen
    batches = torch.zeros((1, tgtlen), dtype=input_values.dtype)
    batches[:, :wavlen] = input_values
    batches = batches.view(-1, maxlen)
    return batches

class WavLMChunk10LayerAvgExtractor:
    def __init__(self, device: str):
        self.device = device
        self.target_sr = int(CFG["target_sr"])
        self.chunk_len = int(float(CFG["chunk_sec"]) * self.target_sr)

        self.dir_last12 = Path(CFG["output_root"]) / "wavlm_large_chunk10_avg_last12"
        self.dir_all24 = Path(CFG["output_root"]) / "wavlm_large_chunk10_avg_all24"
        if CFG["export_avg_last12"]:
            ensure_dir(self.dir_last12 / "frame")
        if CFG["export_avg_all24"]:
            ensure_dir(self.dir_all24 / "frame")

        use_fp16 = bool(CFG["use_fp16"]) and device != "cpu" and torch.cuda.is_available()
        dtype = torch.float16 if use_fp16 else torch.float32

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(CFG["wavlm_id"])
        self.model = WavLMModel.from_pretrained(CFG["wavlm_id"], torch_dtype=dtype).to(device).eval()

    @torch.no_grad()
    def extract(self, wav_path: str) -> Optional[Dict[str, np.ndarray]]:
        wav_1d = load_audio_1d(
            wav_path=wav_path,
            target_sr=self.target_sr,
            do_zmuv=bool(CFG["do_zmuv"]),
            remove_dc=bool(CFG["remove_dc"]),
            max_sec=CFG["max_sec"],
        )
        if wav_1d is None:
            return None

        if wav_1d.numel() < int(CFG["min_samples"]):
            out: Dict[str, np.ndarray] = {}
            if CFG["export_avg_last12"]:
                out["avg_last12"] = np.zeros((1, 1024), dtype=np.float32)
            if CFG["export_avg_all24"]:
                out["avg_all24"] = np.zeros((1, 1024), dtype=np.float32)
            return out

        inputs = self.processor(wav_1d.cpu().numpy(), sampling_rate=self.target_sr, return_tensors="pt")
        input_values = inputs.input_values  # [1, T]
        input_values = split_into_batch(input_values, maxlen=self.chunk_len)  # [B, 10s]
        input_values = input_values.to(self.device)
        if self.model.dtype == torch.float16:
            input_values = input_values.half()

        hidden_states = self.model(input_values, output_hidden_states=True).hidden_states
        if hidden_states is None or len(hidden_states) <= 1:
            last = self.model(input_values).last_hidden_state  # [B, T, D]
            feat = last.reshape(-1, last.shape[-1]).detach().cpu().numpy().astype(np.float32)
            out: Dict[str, np.ndarray] = {}
            if CFG["export_avg_last12"]:
                out["avg_last12"] = feat
            if CFG["export_avg_all24"]:
                out["avg_all24"] = feat
            return out

        layers = hidden_states[1:]  # skip conv/embedding stage, keep transformer layers only
        out: Dict[str, np.ndarray] = {}

        if CFG["export_avg_last12"]:
            sel = layers[-12:] if len(layers) >= 12 else layers
            acc = None
            for h in sel:
                h32 = h.float()
                acc = h32 if acc is None else (acc + h32)
            feat_last12 = (acc / float(len(sel)))  # [B, T, D]
            feat_last12 = feat_last12.reshape(-1, feat_last12.shape[-1])
            out["avg_last12"] = feat_last12.detach().cpu().numpy().astype(np.float32)

        if CFG["export_avg_all24"]:
            acc = None
            for h in layers:
                h32 = h.float()
                acc = h32 if acc is None else (acc + h32)
            feat_all24 = (acc / float(len(layers)))  # [B, T, D]
            feat_all24 = feat_all24.reshape(-1, feat_all24.shape[-1])
            out["avg_all24"] = feat_all24.detach().cpu().numpy().astype(np.float32)

        return out

def save_frame(base_dir: Path, uid: str, frame_arr: np.ndarray):
    ensure_dir(base_dir / "frame")
    np.save(base_dir / "frame" / f"{uid}.npy", frame_arr.astype(np.float32))

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
    records = list(iter_jsonl(CFG["manifest_jsonl"]))
    out_root = Path(CFG["output_root"])
    ensure_dir(out_root)

    print(f"Loaded {len(records)} utterances")
    print(f"Device: {CFG['device']}")
    print("Extractor mode: chunk10 + avg_last12/avg_all24 + frame-only")

    failures = []
    extractor = None

    try:
        extractor = WavLMChunk10LayerAvgExtractor(CFG["device"])

        for rec in tqdm(records, desc="wavlm_chunk10_extract"):
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

            need_last12 = bool(CFG["export_avg_last12"]) and not (
                CFG["skip_if_exists_and_ok"] and need_skip(extractor.dir_last12, uid)
            )
            need_all24 = bool(CFG["export_avg_all24"]) and not (
                CFG["skip_if_exists_and_ok"] and need_skip(extractor.dir_all24, uid)
            )

            if not (need_last12 or need_all24):
                continue

            try:
                feats = extractor.extract(wav_path)
                if feats is None:
                    failures.append({"uid": uid, "error": "extract_none", "audio_path": wav_path})
                    continue

                if need_last12 and "avg_last12" in feats:
                    save_frame(extractor.dir_last12, uid, feats["avg_last12"])
                if need_all24 and "avg_all24" in feats:
                    save_frame(extractor.dir_all24, uid, feats["avg_all24"])
            except Exception as e:
                failures.append({"uid": uid, "error": repr(e), "audio_path": wav_path})

    finally:
        cleanup(extractor)

    if failures:
        fail_path = out_root / CFG["failure_jsonl"]
        with open(fail_path, "w", encoding="utf-8") as f:
            for row in failures:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[WARN] failures saved to {fail_path}")

    print("Done.")
    if CFG["export_avg_last12"]:
        print("feat_root for training (last12):", extractor.dir_last12 / "frame")
    if CFG["export_avg_all24"]:
        print("feat_root for training (all24): ", extractor.dir_all24 / "frame")

if __name__ == "__main__":
    main()
