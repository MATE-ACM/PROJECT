#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract HSEmotion features from aligned faces, with video fallback when needed."""

from __future__ import annotations
import os, json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

CFG = {
    'manifest_jsonl': 'data/manifest/manifest.jsonl',
    'openface_face_root': 'data/features/openface_face',
    'seq_output_root': 'data/features/video_hsemotion_seq_npz',
    'utt_output_root': 'data/features/video_hsemotion_mean_npy',

    'model_name': 'enet_b2_7',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'target_fps': 10,
    'max_frames': 300,
    'face_size': 224,
    'min_face_prob': 0.90,
    'batch_size': 32,
    'overwrite': False,

    'bad_uids': {
        'train_dia125_utt3',
        'val_dia110_utt7',
    },

    'fail_jsonl': '_failures_hsemotion.jsonl',
}

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def iter_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def center_crop_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h-s)//2, (w-s)//2
    return img[y0:y0+s, x0:x0+s]

def crop_with_box(img, box, margin=0.15):
    h, w = img.shape[:2]
    x1,y1,x2,y2 = box
    bw, bh = x2-x1, y2-y1
    x1 = max(0, int(x1 - margin*bw)); y1 = max(0, int(y1 - margin*bh))
    x2 = min(w, int(x2 + margin*bw)); y2 = min(h, int(y2 + margin*bh))
    return img[y1:y2, x1:x2]

def read_aligned_npy(root: Path, uid: str):
    p = root / uid / f'{uid}.npy'
    if not p.exists():
        return None
    try:
        arr = np.load(p, allow_pickle=False)
    except Exception:
        return None
    if not isinstance(arr, np.ndarray):
        return None
    if arr.size == 0 or len(arr) == 0:
        return None
    return arr

def sample_video_frames(video_path: str, target_fps: int, max_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if not orig_fps or np.isnan(orig_fps) or orig_fps <= 1e-3:
        orig_fps = 25.0
    stride = max(1, int(round(orig_fps / target_fps)))
    frames = []
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames

def build_face_imgs_from_video(frames, mtcnn):
    boxes_list, probs_list = mtcnn.detect(frames)
    face_imgs, valid_mask = [], []
    for img, b, p in zip(frames, boxes_list, probs_list):
        use_face = False
        if b is not None and p is not None and len(p) > 0:
            best = int(np.argmax(p))
            if float(p[best]) >= CFG['min_face_prob']:
                face = crop_with_box(img, b[best]); use_face = True
        if not use_face:
            face = center_crop_square(img)
            valid_mask.append(0)
        else:
            valid_mask.append(1)
        face = cv2.resize(face, (CFG['face_size'], CFG['face_size']))
        face_imgs.append(face)
    return face_imgs, valid_mask

@torch.no_grad()
def main():
    ensure_dir(CFG['seq_output_root'])
    ensure_dir(CFG['utt_output_root'])

    mtcnn = MTCNN(keep_all=True, device=CFG['device'])
    fer = HSEmotionRecognizer(model_name=CFG['model_name'], device=CFG['device'])

    fails = []
    records = list(iter_jsonl(CFG['manifest_jsonl']))

    for rec in tqdm(records, desc='hsemotion'):
        uid = str(rec.get('uid') or rec.get('utt_id') or '').strip()
        if not uid:
            continue
        if uid in CFG['bad_uids']:
            fails.append({'uid': uid, 'error': 'skip_bad_uid'})
            continue

        seq_path = Path(CFG['seq_output_root']) / f'{uid}.npz'
        utt_path = Path(CFG['utt_output_root']) / f'{uid}.npy'
        if not CFG['overwrite'] and seq_path.exists() and utt_path.exists():
            continue

        frames = read_aligned_npy(Path(CFG['openface_face_root']), uid)

        if frames is None:
            video_path = str(rec.get('video_path') or '')
            if not video_path or not os.path.exists(video_path):
                fails.append({'uid': uid, 'error': 'missing_video'})
                continue

            frames = sample_video_frames(video_path, CFG['target_fps'], CFG['max_frames'])
            if not frames:
                fails.append({'uid': uid, 'error': 'no_frames'})
                continue

            face_imgs, valid_mask = build_face_imgs_from_video(frames, mtcnn)
        else:
            face_imgs = []
            for x in frames:
                if x is None:
                    continue
                if isinstance(x, np.ndarray) and x.size == 0:
                    continue
                if x.ndim == 3:
                    face_imgs.append(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
            valid_mask = [1] * len(face_imgs)

        if len(face_imgs) == 0:
            fails.append({'uid': uid, 'error': 'empty_face_imgs'})
            continue

        all_feats, all_logits = [], []
        bs = int(CFG['batch_size'])
        for i in range(0, len(face_imgs), bs):
            batch = face_imgs[i:i+bs]
            if len(batch) == 0:
                continue
            try:
                b_feats = fer.extract_multi_features(batch)
                _, b_logits = fer.predict_multi_emotions(batch, logits=True)
            except Exception as e:
                fails.append({'uid': uid, 'error': f'hsemotion_batch_failed:{repr(e)}'})
                continue
            if b_feats is None or len(b_feats) == 0:
                continue
            if b_logits is None or len(b_logits) == 0:
                continue
            all_feats.append(b_feats)
            all_logits.append(b_logits)

        if len(all_feats) == 0 or len(all_logits) == 0:
            fails.append({'uid': uid, 'error': 'empty_hsemotion_outputs'})
            continue

        try:
            feats = np.concatenate(all_feats, axis=0).astype(np.float32)
            logits = np.concatenate(all_logits, axis=0).astype(np.float32)
            np.savez_compressed(
                seq_path,
                x=feats,
                emo_logits=logits,
                face_detected=np.asarray(valid_mask, np.int8),
                fps=np.asarray([CFG['target_fps']], np.float32)
            )
            np.save(utt_path, feats.mean(axis=0).astype(np.float32))
        except Exception as e:
            fails.append({'uid': uid, 'error': f'save_failed:{repr(e)}'})
            continue

    if fails:
        with open(Path(CFG['seq_output_root']) / CFG['fail_jsonl'], 'w', encoding='utf-8') as f:
            for x in fails:
                f.write(json.dumps(x, ensure_ascii=False) + '\n')

    print('Done.')

if __name__ == '__main__':
    main()
