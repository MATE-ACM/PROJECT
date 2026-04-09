#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import os
import re
import json
import math
import subprocess
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
import torch
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer


_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load
# ==========================================


# ================= ⚙️ 配置区域（只改这里） =================
CFG = {
    "MANIFEST_PATH": r"patch_missing_by_feature_video.jsonl",

    # --- (可选) 先切 utterance mp4 ---
    "RUN_SEGMENT_MP4": True,
    "FFMPEG": r"ffmpeg.exe",  # 或写绝对路径 r"ffmpeg.exe"
    "OVERWRITE_SEGMENT": False,
    "EMIT_VIDEO_ROOT": r"iemocap_utt_video_mp4",

    # 切分时就做 half-crop（推荐 True；这样后面提 hsemotion 就不用再裁半边）
    "SEGMENT_CROP_HALF": True,

    # --- HSEmotion ---
    "RUN_HSEMOTION": True,
    "VIDEO_KEY_CANDIDATES": ["video_path", "video", "mp4_path", "path_video", "dialog_video_path"],
    "UTT_KEY_CANDIDATES": ["utt_id", "uid"],

    "OUT_ROOT": r"video_hsemotion_enet_b2_7_seq_npz",

    "target_fps": 10,
    "max_frames": 300,
    "face_size": 224,
    "min_face_prob": 0.90,

    # 防止显存爆炸
    "batch_size": 32,

    # HSEmotion 模型名
    "model_name": "enet_b2_7",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "overwrite": False,

    # 如果你在切分阶段没 half-crop（SEGMENT_CROP_HALF=False），
    # 那这里可以开 True 来“逐帧裁半边”
    "HSEMOTION_APPLY_HALF_CROP": False,
}
# ============================================================


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_manifest_to_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)
    return df


def pick_first_present(row: Any, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in row and isinstance(row[k], str) and row[k]:
            return row[k]
    # itertuples 模式
    for k in keys:
        if hasattr(row, k):
            v = getattr(row, k)
            if isinstance(v, str) and v:
                return v
    return None


def infer_half_from_utt_id(utt_id: str) -> Optional[str]:
 
    if not isinstance(utt_id, str) or len(utt_id) < 6:
        return None
    try:
        left_gender = utt_id[5]
        target_gender = utt_id[-4]
        if left_gender not in ("M", "F") or target_gender not in ("M", "F"):
            return None
        return "left" if left_gender == target_gender else "right"
    except Exception:
        return None


def ffmpeg_segment_video(src: str, start_sec: float, end_sec: float, out_mp4: str, crop_half: Optional[str]) -> bool:
    ensure_dir(os.path.dirname(out_mp4))

    vf = None
    if crop_half == "left":
        vf = "crop=iw/2:ih:0:0"
    elif crop_half == "right":
        vf = "crop=iw/2:ih:iw/2:0"

    cmd = [
        CFG["FFMPEG"],
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{start_sec:.3f}",
        "-to", f"{end_sec:.3f}",
        "-i", src,
        "-map", "0:v:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
    ]
    if vf:
        cmd += ["-vf", vf]
    cmd += [out_mp4, "-y"]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def run_segment_mp4(df: pd.DataFrame) -> None:
    need_cols = {"start_sec", "end_sec"}
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"RUN_SEGMENT_MP4=True but manifest missing column: {c}")

    failures = []
    done = 0
    skip = 0

    for r in tqdm(df.to_dict(orient="records"), total=len(df), desc="segment_mp4"):
        utt_id = r.get("utt_id") or r.get("uid")
        if not utt_id:
            continue

        src = r.get("dialog_video_path") or r.get("video_src")
        if not src or not os.path.exists(src):
            failures.append({"utt_id": utt_id, "err": "missing_dialog_video", "src": src})
            continue

        out_mp4 = r.get("video_path")
        if not isinstance(out_mp4, str) or not out_mp4:
            out_mp4 = os.path.join(CFG["EMIT_VIDEO_ROOT"], f"{utt_id}.mp4")

        if (not CFG["OVERWRITE_SEGMENT"]) and os.path.exists(out_mp4):
            skip += 1
            continue

        st = float(r.get("start_sec", 0.0))
        ed = float(r.get("end_sec", 0.0))
        if ed <= st:
            failures.append({"utt_id": utt_id, "err": "bad_time", "start_sec": st, "end_sec": ed})
            continue

        crop_half = infer_half_from_utt_id(str(utt_id)) if CFG["SEGMENT_CROP_HALF"] else None

        ok = ffmpeg_segment_video(src, st, ed, out_mp4, crop_half=crop_half)
        if ok:
            done += 1
        else:
            failures.append({"utt_id": utt_id, "err": "ffmpeg_failed", "src": src})

    print(f"[segment] done={done} skip={skip} failures={len(failures)}")
    if failures:
        p = os.path.join(CFG["OUT_ROOT"], "_failures_segment_mp4.jsonl")
        ensure_dir(CFG["OUT_ROOT"])
        with open(p, "w", encoding="utf-8") as f:
            for x in failures:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"[segment] failures -> {p}")


def center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def crop_with_box(img: np.ndarray, box, margin: float = 0.15) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - margin * bw))
    y1 = max(0, int(y1 - margin * bh))
    x2 = min(w, int(x2 + margin * bw))
    y2 = min(h, int(y2 + margin * bh))
    return img[y1:y2, x1:x2]


def crop_half_frame(frame_rgb: np.ndarray, half: str) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    if half == "left":
        return frame_rgb[:, : w // 2, :]
    else:
        return frame_rgb[:, w // 2:, :]


@torch.no_grad()
def run_hsemotion(df: pd.DataFrame) -> None:
    ensure_dir(CFG["OUT_ROOT"])
    device = CFG["device"]

    print(f"[INFO] Loading models on {device}...")
    mtcnn = MTCNN(keep_all=True, device=device)

    print("[INFO] Initializing HSEmotion (patched torch.load)...")
    fer = HSEmotionRecognizer(model_name=CFG["model_name"], device=device)

    failures = []

    items = df.to_dict(orient="records")
    for it in tqdm(items, desc="extract_hsemotion"):
        utt = pick_first_present(it, CFG["UTT_KEY_CANDIDATES"])
        if not utt:
            continue
        utt = str(utt)

        vpath = pick_first_present(it, CFG["VIDEO_KEY_CANDIDATES"])
        if not vpath or (not os.path.exists(vpath)):
            failures.append({"utt_id": utt, "error": "video_not_found"})
            continue

        out_path = os.path.join(CFG["OUT_ROOT"], f"{utt}.npz")
        if (not CFG["overwrite"]) and os.path.exists(out_path):
            continue

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            failures.append({"utt_id": utt, "error": "cannot_open"})
            continue

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if not orig_fps or np.isnan(orig_fps) or orig_fps <= 1e-3:
            orig_fps = 25.0

        stride = max(1, int(round(orig_fps / CFG["target_fps"])))
        frames = []
        frame_idx = []

        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                if CFG["HSEMOTION_APPLY_HALF_CROP"]:
                    half = infer_half_from_utt_id(utt)
                    if half in ("left", "right"):
                        rgb = crop_half_frame(rgb, half)

                frames.append(rgb)
                frame_idx.append(idx)
                if len(frames) >= CFG["max_frames"]:
                    break
            idx += 1
        cap.release()

        if not frames:
            failures.append({"utt_id": utt, "error": "no_frames"})
            continue

        # 1) 人脸检测（整段一起 detect，速度更快）
        try:
            boxes_list, probs_list = mtcnn.detect(frames)
        except RuntimeError:
            boxes_list = [None] * len(frames)
            probs_list = [None] * len(frames)

        face_imgs = []
        valid_mask = []

        for img, b, p in zip(frames, boxes_list, probs_list):
            use_face = False
            if b is not None and p is not None:
                best = int(np.argmax(p))
                if float(p[best]) >= CFG["min_face_prob"]:
                    face = crop_with_box(img, b[best])
                    use_face = True

            if not use_face:
                face = center_crop_square(img)
                valid_mask.append(0)
            else:
                valid_mask.append(1)

            face = cv2.resize(face, (CFG["face_size"], CFG["face_size"]), interpolation=cv2.INTER_LINEAR)
            face_imgs.append(face)

        # 2) 分批提取 Feature + Logits
        all_feats = []
        all_logits = []

        bs = int(CFG["batch_size"])
        for i in range(0, len(face_imgs), bs):
            batch_faces = face_imgs[i: i + bs]
            b_feats = fer.extract_multi_features(batch_faces)               # [B, D]
            _, b_logits = fer.predict_multi_emotions(batch_faces, logits=True)  # [B, K]
            all_feats.append(b_feats)
            all_logits.append(b_logits)

        if all_feats:
            feats = np.concatenate(all_feats, axis=0).astype(np.float32)
            emo_logits = np.concatenate(all_logits, axis=0).astype(np.float32)
        else:
            feats = np.zeros((0, 1280), dtype=np.float32)
            emo_logits = np.zeros((0, 8), dtype=np.float32)

        np.savez_compressed(
            out_path,
            x=feats,
            emo_logits=emo_logits,
            fps=np.array([CFG["target_fps"]], dtype=np.float32),
            frame_idx=np.array(frame_idx, dtype=np.int32),
            face_detected=np.array(valid_mask, dtype=np.int8),
        )

    if failures:
        p = os.path.join(CFG["OUT_ROOT"], "_failures_hsemotion.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for x in failures:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"[WARN] {len(failures)} failures -> {p}")
    else:
        print("[INFO] No failures.")


def main():
    df = read_manifest_to_df(CFG["MANIFEST_PATH"])
    print(f">>> Manifest loaded: {len(df)} rows")

    if CFG["RUN_SEGMENT_MP4"]:
        run_segment_mp4(df)

    if CFG["RUN_HSEMOTION"]:
        run_hsemotion(df)

    print(" All done.")


if __name__ == "__main__":
    main()
