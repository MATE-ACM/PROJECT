# scripts/extract_video_hsemotion_npz_click.py
from __future__ import annotations
import os, json
import numpy as np
from tqdm import tqdm

import cv2
import torch
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer


import torch.serialization
from functools import partial

# 保存原始的 torch.load
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    # 强制将 weights_only 设为 False，允许加载旧版模型
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


# 覆盖系统的 torch.load
torch.load = _patched_torch_load
# ==========================================



CFG = {
    "manifest_jsonl": r"manifest.jsonl",
    "video_key_candidates": ["video_path", "video", "mp4_path", "path_video"],
    "utt_key": "utt_id",

    "out_root": r"hsemotion_enet_b2_7_seq_npz",

    "target_fps": 10,
    "max_frames": 300,
    "face_size": 224,
    "min_face_prob": 0.90,

    # 防止显存爆炸的关键参数
    "batch_size": 32,

    # HSEmotion 模型名
    "model_name": "enet_b2_7",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "overwrite": False,
}


def _read_manifest(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            items.append(json.loads(line))
    return items


def _pick_video_path(item: dict, keys: list[str]) -> str:
    for k in keys:
        if k in item and item[k]: return item[k]
    raise KeyError(f"Cannot find video path. Tried={keys}")


def _center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def _crop_with_box(img: np.ndarray, box, margin: float = 0.15) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - margin * bw))
    y1 = max(0, int(y1 - margin * bh))
    x2 = min(w, int(x2 + margin * bw))
    y2 = min(h, int(y2 + margin * bh))
    return img[y1:y2, x1:x2]


@torch.no_grad()
def main():
    os.makedirs(CFG["out_root"], exist_ok=True)
    items = _read_manifest(CFG["manifest_jsonl"])
    device = CFG["device"]

    print(f"[INFO] Loading models on {device}...")

    # MTCNN
    mtcnn = MTCNN(keep_all=True, device=device)


    print("[INFO] Initializing HSEmotion (patched for PyTorch 2.6+)...")
    fer = HSEmotionRecognizer(model_name=CFG["model_name"], device=device)

    failures = []

    # 打印一下显存保护策略
    print(f"[INFO] Batch processing enabled (batch_size={CFG['batch_size']}) to prevent OOM.")

    for it in tqdm(items, desc="extract_hsemotion"):
        utt = str(it[CFG["utt_key"]])
        vpath = _pick_video_path(it, CFG["video_key_candidates"])

        out_path = os.path.join(CFG["out_root"], f"{utt}.npz")
        if (not CFG["overwrite"]) and os.path.exists(out_path):
            continue

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            failures.append({"utt_id": utt, "error": "cannot_open"})
            continue

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if not orig_fps or np.isnan(orig_fps) or orig_fps <= 1e-3: orig_fps = 25.0

        stride = max(1, int(round(orig_fps / CFG["target_fps"])))
        frames = []
        frame_idx = []

        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok: break
            if idx % stride == 0:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                frame_idx.append(idx)
                if len(frames) >= CFG["max_frames"]: break
            idx += 1
        cap.release()

        if not frames:
            failures.append({"utt_id": utt, "error": "no_frames"})
            continue

        # 1) 人脸检测
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
                    face = _crop_with_box(img, b[best])
                    use_face = True

            if not use_face:
                face = _center_crop_square(img)
                valid_mask.append(0)
            else:
                valid_mask.append(1)

            face = cv2.resize(face, (CFG["face_size"], CFG["face_size"]), interpolation=cv2.INTER_LINEAR)
            face_imgs.append(face)

        # 2) 分批提取 Feature + Logits
        all_feats = []
        all_logits = []

        bs = CFG["batch_size"]
        for i in range(0, len(face_imgs), bs):
            batch_faces = face_imgs[i: i + bs]

            b_feats = fer.extract_multi_features(batch_faces)
            _, b_logits = fer.predict_multi_emotions(batch_faces, logits=True)

            all_feats.append(b_feats)
            all_logits.append(b_logits)

        if all_feats:
            feats = np.concatenate(all_feats, axis=0)  # [T, D]
            emo_logits = np.concatenate(all_logits, axis=0)  # [T, 7/8]
        else:
            feats = np.zeros((0, 1280), dtype=np.float32)
            emo_logits = np.zeros((0, 8), dtype=np.float32)

        np.savez_compressed(
            out_path,
            x=feats.astype(np.float32),
            emo_logits=emo_logits.astype(np.float32),
            fps=np.array([CFG["target_fps"]], dtype=np.float32),
            frame_idx=np.array(frame_idx, dtype=np.int32),
            face_detected=np.array(valid_mask, dtype=np.int8),
        )

    if failures:
        print(f"[WARN] {len(failures)} failures.")


if __name__ == "__main__":
    main()