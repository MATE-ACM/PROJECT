from __future__ import annotations

"""
OpenFace CSV -> per-utt NPZ sequence files (schema-consistent, fusion-friendly)

Output:
  <out_dir>/<utt_id>.npz
    x: [T, D] float32
    mask: [T] uint8   (1=valid frame, 0=invalid)
    frame: [T] int32 (optional, if exists else arange)
    timestamp: [T] float32 (optional, if exists else -1)
    success: [T] uint8 (optional)
    confidence: [T] float32 (optional)
    cols: [D] object array (global fixed schema)
Also writes:
  <out_dir>/_schema_cols.txt
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

# =========================
# ✅ 点击改这里
# =========================
CFG = {
    "input_path": r"FacialTracking_Actors_01-24",   # dir or single csv
    "out_dir": r"video",

    # [] -> auto
    # or manual list
    "cols": [],

    # schema policy:
    #  - "first": use first file's auto cols as global schema
    #  - "intersection": only keep cols present in ALL files (最稳但可能丢信息)
    #  - "union": keep union of cols (维度可能更大，但信息最全)
    "schema_policy": "first",

    # valid frame rule: success>=0.5 AND confidence>=thr (if exists)
    "confidence_thr": 0.80,

    # optional: add delta (first difference) features (会让 D 变成 2D)
    "add_delta": False,
}
# =========================


def _auto_select_cols(df: pd.DataFrame) -> List[str]:
    df.columns = df.columns.str.strip()

    cols: List[str] = []
    au_r = sorted([c for c in df.columns if c.startswith("AU") and c.endswith("_r")])
    au_c = sorted([c for c in df.columns if c.startswith("AU") and c.endswith("_c")])

    # 这里给你一个更“情绪友好”的默认：同时保留 intensity + presence（可解释更强）
    # 如果你不想加维度，可改回只用 au_r 或只用 au_c
    if au_r:
        cols += au_r
    if au_c:
        cols += au_c

    for c in ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]:
        if c in df.columns:
            cols.append(c)

    if "gaze_angle_x" in df.columns and "gaze_angle_y" in df.columns:
        cols += ["gaze_angle_x", "gaze_angle_y"]
    else:
        for c in ["gaze_0_x", "gaze_0_y", "gaze_1_x", "gaze_1_y"]:
            if c in df.columns:
                cols.append(c)

    # fallback：保留所有数值列（排除 meta）
    if not cols:
        exclude = {"frame", "timestamp", "confidence", "success", "face_id"}
        cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    # 去重但保持顺序
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _scan_schema(csv_paths: List[Path], cols_cfg: List[str]) -> List[str]:
    policy = str(CFG["schema_policy"]).lower()

    schema_cols: Optional[List[str]] = None

    for i, p in enumerate(csv_paths):
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        cols = cols_cfg if cols_cfg else _auto_select_cols(df)

        if schema_cols is None:
            schema_cols = cols
            if policy == "first":
                break
        else:
            if policy == "intersection":
                schema_cols = [c for c in schema_cols if c in set(cols)]
            elif policy == "union":
                s = set(schema_cols)
                for c in cols:
                    if c not in s:
                        schema_cols.append(c)
                        s.add(c)

    assert schema_cols is not None
    return schema_cols


def _process_one_df(df: pd.DataFrame, schema_cols: List[str]) -> Dict[str, np.ndarray]:
    df.columns = df.columns.str.strip()

    if "frame" in df.columns:
        df = df.sort_values("frame")

    # frame / timestamp
    if "frame" in df.columns:
        frame = df["frame"].to_numpy(dtype=np.int32)
    else:
        frame = np.arange(len(df), dtype=np.int32)

    if "timestamp" in df.columns:
        timestamp = df["timestamp"].to_numpy(dtype=np.float32)
    else:
        timestamp = np.full((len(df),), -1.0, dtype=np.float32)

    # success/confidence & mask
    success = None
    confidence = None
    mask = np.ones((len(df),), dtype=np.uint8)

    if "success" in df.columns:
        success = df["success"].astype(float).to_numpy()
        mask &= (success >= 0.5).astype(np.uint8)
    if "confidence" in df.columns:
        confidence = df["confidence"].astype(float).to_numpy()
        mask &= (confidence >= float(CFG["confidence_thr"])).astype(np.uint8)

    # build x with fixed schema, missing -> 0
    x = np.zeros((len(df), len(schema_cols)), dtype=np.float32)
    for j, c in enumerate(schema_cols):
        if c in df.columns:
            col = df[c].to_numpy(dtype=np.float32)
            x[:, j] = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)

    # invalid frames -> zero out (仍保留 mask 供模型/融合用)
    if mask is not None:
        x[mask == 0] = 0.0

    # optional delta features (对动态情绪有时有帮助)
    if bool(CFG["add_delta"]):
        dx = np.zeros_like(x)
        dx[1:] = x[1:] - x[:-1]
        x = np.concatenate([x, dx], axis=1)

    if x.shape[0] == 0:
        x = np.zeros((1, len(schema_cols)), dtype=np.float32)
        mask = np.ones((1,), dtype=np.uint8)
        frame = np.zeros((1,), dtype=np.int32)
        timestamp = np.full((1,), -1.0, dtype=np.float32)

    out = {
        "x": x.astype(np.float32),
        "mask": mask.astype(np.uint8),
        "frame": frame.astype(np.int32),
        "timestamp": timestamp.astype(np.float32),
    }
    if success is not None:
        out["success"] = success.astype(np.uint8)
    if confidence is not None:
        out["confidence"] = confidence.astype(np.float32)
    return out


def main():
    in_path = Path(CFG["input_path"])
    out_dir = Path(CFG["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    cols_cfg = CFG["cols"]

    if in_path.is_file():
        # single csv mode: requires utt_id
        df_all = pd.read_csv(in_path)
        if "utt_id" not in df_all.columns:
            raise KeyError("Single-CSV mode requires 'utt_id' column.")
        # schema from full df (safer)
        schema_cols = cols_cfg if cols_cfg else _auto_select_cols(df_all)
        (out_dir / "_schema_cols.txt").write_text("\n".join(schema_cols), encoding="utf-8")

        n = 0
        for utt_id, g in df_all.groupby("utt_id"):
            pack = _process_one_df(g, schema_cols)
            np.savez_compressed(out_dir / f"{utt_id}.npz", cols=np.array(schema_cols, dtype=object), **pack)
            n += 1
        print(f"[OK] Processed {n} utterances. Saved to: {out_dir}")
        return

    # dir mode
    csvs = sorted(in_path.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No .csv found in dir: {in_path}")

    schema_cols = _scan_schema(csvs, cols_cfg)
    (out_dir / "_schema_cols.txt").write_text("\n".join(schema_cols), encoding="utf-8")
    print(f"[INFO] schema_policy={CFG['schema_policy']}  D={len(schema_cols)}")

    for p in csvs:
        utt_id = p.stem
        df = pd.read_csv(p)
        pack = _process_one_df(df, schema_cols)
        np.savez_compressed(out_dir / f"{utt_id}.npz", cols=np.array(schema_cols, dtype=object), **pack)

    print(f"[OK] Processed {len(csvs)} files. Saved to: {out_dir}")


if __name__ == "__main__":
    main()
