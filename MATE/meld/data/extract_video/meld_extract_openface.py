#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract OpenFace outputs and keep only the files used downstream.

Saved outputs:
- openface_face/<uid>/<uid>.npy
- openface_csv/<uid>.csv
"""
from __future__ import annotations

import os
import shutil
import glob
import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

CFG = {
    "subvideo_root": "data/subvideo",

    "features_root": "data/features",

    # OpenFace FeatureExtraction.exe
    "openface_exe": "path/to/OpenFace/FeatureExtraction",

    "overwrite": False,

    "reuse_legacy_openface_all": True,

    "cleanup_legacy_openface_all_after_extract": True,

    "cleanup_redundant_root_dirs": False,

    "allow_missing_face_npy": False,

    "log_jsonl": "_openface_slim_failures.jsonl",

    "summary_json": "_openface_slim_summary.json",
}

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def is_npy_ok(path: Path) -> bool:
    try:
        if not path.exists():
            return False
        arr = np.load(path, allow_pickle=False)
        return isinstance(arr, np.ndarray) and arr.ndim == 4 and arr.shape[0] >= 1 and arr.shape[-1] in (3, 4)
    except Exception:
        return False

def is_csv_ok(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size < 32:
            return False
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(4096).lower()
        return ("frame" in head) and (("confidence" in head) or ("success" in head))
    except Exception:
        return False

def find_primary_csv(root: Path) -> Optional[Path]:
    candidates = sorted(root.glob("*.csv"))
    if not candidates:
        return None
    return candidates[0]

def collect_aligned_frames(tmp_out_root: Path) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for dir_path in sorted(glob.glob(str(tmp_out_root / "*_aligned"))):
        frame_names = sorted(os.listdir(dir_path))
        for frame_name in frame_names:
            frame_path = os.path.join(dir_path, frame_name)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames.append(frame)
    return frames

def write_face_npy_from_outdir(openface_out_dir: Path, uid: str, face_uid_dir: Path) -> bool:
    frames = collect_aligned_frames(openface_out_dir)
    if len(frames) == 0:
        return False
    ensure_dir(face_uid_dir)
    save_path = face_uid_dir / f"{uid}.npy"
    np.save(save_path, np.asarray(frames, dtype=np.uint8))
    return True

def copy_csv_from_outdir(openface_out_dir: Path, uid: str, csv_root: Path) -> bool:
    src = find_primary_csv(openface_out_dir)
    if src is None:
        return False
    ensure_dir(csv_root)
    dst = csv_root / f"{uid}.csv"
    shutil.copy2(src, dst)
    return True

def run_openface_to_temp(video_path: Path, openface_exe: str) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="of_tmp_"))
    cmd = [str(openface_exe), "-f", str(video_path), "-out_dir", str(tmp_dir)]
    import subprocess
    subprocess.run(cmd, check=True)
    return tmp_dir

def remove_path(p: Path):
    try:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink(missing_ok=True)
    except Exception:
        pass

def maybe_cleanup_redundant_roots(features_root: Path):
    if not CFG["cleanup_redundant_root_dirs"]:
        return
    for name in ["openface_all", "openface_hog", "openface_pose"]:
        p = features_root / name
        if p.exists():
            print(f"[cleanup-root] removing {p}")
            remove_path(p)

def process_one(uid: str, video_path: Path, face_root: Path, csv_root: Path, legacy_all_root: Path,
                failures: list[dict], stats: dict):
    face_uid_dir = face_root / uid
    face_npy = face_uid_dir / f"{uid}.npy"
    csv_path = csv_root / f"{uid}.csv"

    face_ok = is_npy_ok(face_npy)
    csv_ok = is_csv_ok(csv_path)

    if face_ok and csv_ok and not CFG["overwrite"]:
        stats["skipped_complete"] += 1
        return

    legacy_dir = legacy_all_root / uid
    if CFG["reuse_legacy_openface_all"] and legacy_dir.exists():
        reused_any = False
        if (CFG["overwrite"] or not csv_ok):
            ok = copy_csv_from_outdir(legacy_dir, uid, csv_root)
            csv_ok = is_csv_ok(csv_path)
            reused_any = reused_any or ok
        if (CFG["overwrite"] or not face_ok):
            ok = write_face_npy_from_outdir(legacy_dir, uid, face_uid_dir)
            face_ok = is_npy_ok(face_npy)
            reused_any = reused_any or ok

        if reused_any:
            stats["reused_legacy"] += 1

        if csv_ok and (face_ok or CFG["allow_missing_face_npy"]):
            if CFG["cleanup_legacy_openface_all_after_extract"]:
                remove_path(legacy_dir)
            stats["done_from_legacy"] += 1
            return

    tmp_dir = None
    try:
        tmp_dir = run_openface_to_temp(video_path, CFG["openface_exe"])
        if CFG["overwrite"] or not csv_ok:
            copy_csv_from_outdir(tmp_dir, uid, csv_root)
        if CFG["overwrite"] or not face_ok:
            write_face_npy_from_outdir(tmp_dir, uid, face_uid_dir)

        face_ok = is_npy_ok(face_npy)
        csv_ok = is_csv_ok(csv_path)
        if csv_ok and (face_ok or CFG["allow_missing_face_npy"]):
            stats["done_from_fresh_run"] += 1
        else:
            failures.append({
                "uid": uid,
                "video_path": str(video_path),
                "error": "openface_outputs_incomplete",
                "face_ok": face_ok,
                "csv_ok": csv_ok,
            })
            stats["failed"] += 1
    except Exception as e:
        failures.append({
            "uid": uid,
            "video_path": str(video_path),
            "error": repr(e),
        })
        stats["failed"] += 1
    finally:
        if tmp_dir is not None:
            remove_path(tmp_dir)

def main():
    subvideo_root = Path(CFG["subvideo_root"])
    features_root = Path(CFG["features_root"])
    face_root = features_root / "openface_face"
    csv_root = features_root / "openface_csv"
    legacy_all_root = features_root / "openface_all"

    ensure_dir(face_root)
    ensure_dir(csv_root)

    vids = sorted([p for p in subvideo_root.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}])
    print(f"Find total {len(vids)} videos")

    stats = {
        "total_videos": len(vids),
        "skipped_complete": 0,
        "reused_legacy": 0,
        "done_from_legacy": 0,
        "done_from_fresh_run": 0,
        "failed": 0,
    }
    failures: list[dict] = []

    for i, vid in enumerate(vids, 1):
        uid = vid.stem
        print(f"[{i}/{len(vids)}] {uid}")
        process_one(uid, vid, face_root, csv_root, legacy_all_root, failures, stats)

    maybe_cleanup_redundant_roots(features_root)

    if failures:
        log_path = features_root / CFG["log_jsonl"]
        with open(log_path, "w", encoding="utf-8") as f:
            for row in failures:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[WARN] failures saved to {log_path}")

    summary_path = features_root / CFG["summary_json"]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"face root: {face_root}")
    print(f"csv  root: {csv_root}")

if __name__ == "__main__":
    main()
