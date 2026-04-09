#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations
import os
import json
import shutil
import subprocess
from typing import Optional, Dict, Any, List

import pandas as pd
from tqdm import tqdm


# ====================== 你只需要改这里 ======================
CFG = {
   
    "MANIFEST_JSONL": r"patch_missing_by_feature_openface.jsonl",

    # OpenFace 可执行文件
    "OPENFACE_EXE":r"FeatureExtraction.exe",

    # OpenFace 输出根目录（每个 utt 一个文件夹）
    "OUT_OPENFACE_ROOT": r"openface_csv_all",

    # --------- 自动切 utterance mp4（如果 video_path 不存在）---------
    "AUTO_SEGMENT_MP4_IF_MISSING": True,
    "FFMPEG_EXE": "ffmpeg",  # 或 r"ffmpeg.exe"
    "EMIT_VIDEO_ROOT": r"iemocap_utt_video_mp4",
    "OVERWRITE_SEGMENT_MP4": False,

    # IEMOCAP 双人同框：切 mp4 时做 half-crop（强烈建议 True）
    "SEGMENT_HALF_CROP": True,

    # --------- OpenFace 参数：全特征 CSV 常用开关 -----------
   
    "OPENFACE_ARGS": ["-2Dfp", "-3Dfp", "-pdmparams", "-pose", "-gaze", "-aus", "-hogalign"],

    # 是否覆盖已存在的 utt_id.csv
    "OVERWRITE_OPENFACE": False,

    # 日志
    "PRINT_EVERY_N": 200,
}
# ===========================================================


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_manifest_jsonl(path: str) -> List[Dict[str, Any]]:
    return pd.read_json(path, lines=True).to_dict(orient="records")


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


def ffmpeg_segment_mp4(src_avi: str, start_sec: float, end_sec: float, out_mp4: str, half: Optional[str]) -> bool:
    ensure_dir(os.path.dirname(out_mp4))

    vf = None
    if half == "left":
        vf = "crop=iw/2:ih:0:0"
    elif half == "right":
        vf = "crop=iw/2:ih:iw/2:0"

    cmd = [
        CFG["FFMPEG_EXE"],
        "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_sec:.3f}",
        "-to", f"{end_sec:.3f}",
        "-i", src_avi,
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
        return os.path.exists(out_mp4)
    except subprocess.CalledProcessError:
        return False


def run_openface(video_path: str, utt_id: str) -> Optional[str]:
    """
    运行 OpenFace 并返回生成的 csv 路径（最终保证叫 utt_id.csv）
    """
    out_dir = os.path.join(CFG["OUT_OPENFACE_ROOT"], utt_id)
    ensure_dir(out_dir)

    final_csv = os.path.join(out_dir, f"{utt_id}.csv")
    if (not CFG["OVERWRITE_OPENFACE"]) and os.path.exists(final_csv):
        return final_csv

    if not os.path.exists(CFG["OPENFACE_EXE"]):
        raise FileNotFoundError(f"OPENFACE_EXE not found: {CFG['OPENFACE_EXE']}")

    # -of 会把输出前缀设成 utt_id（大多数 build 会生成 utt_id.csv）
    cmd = [
        CFG["OPENFACE_EXE"],
        "-f", video_path,
        "-out_dir", out_dir,
        "-of", utt_id,
    ] + list(CFG["OPENFACE_ARGS"] or [])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        return None

    # 1) 优先找 utt_id.csv
    if os.path.exists(final_csv):
        return final_csv

    # 2) 兜底：某些 build 会用视频名当 csv 名（这里把第一个 csv 重命名成 utt_id.csv）
    csvs = [fn for fn in os.listdir(out_dir) if fn.lower().endswith(".csv")]
    if not csvs:
        return None
    src_csv = os.path.join(out_dir, csvs[0])
    try:
        shutil.copy(src_csv, final_csv)
        return final_csv
    except Exception:
        return src_csv


def main():
    items = read_manifest_jsonl(CFG["MANIFEST_JSONL"])
    print(f"[INFO] manifest rows = {len(items)}")

    ensure_dir(CFG["OUT_OPENFACE_ROOT"])
    ensure_dir(CFG["EMIT_VIDEO_ROOT"])

    failures = []
    ok = 0
    seg = 0
    skipped = 0

    for i, it in enumerate(tqdm(items, desc="openface_keep_csv")):
        utt_id = str(it.get("utt_id", "")).strip()
        if not utt_id:
            continue

        # 目标 video_path
        video_path = it.get("video_path")
        if not isinstance(video_path, str) or not video_path:
            video_path = os.path.join(CFG["EMIT_VIDEO_ROOT"], f"{utt_id}.mp4")

        # 如果 utt mp4 不存在，自动从对话 avi 切
        if not os.path.exists(video_path):
            if not CFG["AUTO_SEGMENT_MP4_IF_MISSING"]:
                failures.append({"utt_id": utt_id, "error": "video_missing", "video_path": video_path})
                continue

            src_avi = it.get("dialog_video_path")
            st = float(it.get("start_sec", 0.0))
            ed = float(it.get("end_sec", 0.0))

            if not (isinstance(src_avi, str) and os.path.exists(src_avi)):
                failures.append({"utt_id": utt_id, "error": "dialog_video_missing", "dialog_video_path": src_avi})
                continue
            if ed <= st:
                failures.append({"utt_id": utt_id, "error": "bad_time", "start_sec": st, "end_sec": ed})
                continue

            if (not CFG["OVERWRITE_SEGMENT_MP4"]) and os.path.exists(video_path):
                pass
            else:
                half = infer_half_from_utt_id(utt_id) if CFG["SEGMENT_HALF_CROP"] else None
                if not ffmpeg_segment_mp4(src_avi, st, ed, video_path, half):
                    failures.append({"utt_id": utt_id, "error": "ffmpeg_failed", "src": src_avi})
                    continue
                seg += 1

        # 跑 OpenFace（并保留 CSV）
        out_dir = os.path.join(CFG["OUT_OPENFACE_ROOT"], utt_id)
        final_csv = os.path.join(out_dir, f"{utt_id}.csv")
        if (not CFG["OVERWRITE_OPENFACE"]) and os.path.exists(final_csv):
            skipped += 1
            continue

        csv_path = run_openface(video_path, utt_id)
        if csv_path and os.path.exists(csv_path):
            ok += 1
        else:
            failures.append({"utt_id": utt_id, "error": "openface_failed", "video_path": video_path})

        if CFG["PRINT_EVERY_N"] and (i + 1) % int(CFG["PRINT_EVERY_N"]) == 0:
            print(f"[INFO] processed={i+1} ok={ok} skipped={skipped} seg={seg} fail={len(failures)}")

    print(f"[DONE] ok={ok} skipped={skipped} segmented={seg} failures={len(failures)}")

    if failures:
        fp = os.path.join(CFG["OUT_OPENFACE_ROOT"], "_failures_openface_keep_csv.jsonl")
        with open(fp, "w", encoding="utf-8") as f:
            for x in failures:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"[WARN] failures saved: {fp}")


if __name__ == "__main__":
    main()
