from __future__ import annotations

"""
【文件作用】校准入口：读取 val/test logits，拟合 temperature/vector scaling，并输出校准报告与可选 reliability 曲线。

Calibration entrypoint (per expert, per fold).

Designed to be called directly from Python:
    from scripts.calibrate import calibrate_run_dir
    calibrate_run_dir("runs/<exp>/fold_0", num_classes=6, method="temperature")

It reads:
- preds/val_logits.npz  (to fit calibrator)
- preds/test_logits.npz (to evaluate)

It writes:
- calibration/report.json (pre/post: WAR/UAR + ECE/Brier/NLL)
- calibration/<method>.json (calibrator parameters)
- preds/*_logits_calib.npz (calibrated logits for downstream fusion/router)
"""

import os
import json
from typing import Dict, Any, Tuple
import numpy as np

from src.ser.calibration import TemperatureScaling, VectorScaling, ConfidenceIsotonic, ConfidenceLinear, save_calibrator
from src.ser.calibration import summarize_calibration, summarize_quality
from src.ser.metrics import summarize as summarize_cls


def _load_npz(path: str):
    """Load (utt_id, y_true, logits) from npz produced by training."""
    d = np.load(path, allow_pickle=True)
    utt_id = d["utt_id"].tolist()
    y_true = d["y_true"].astype(np.int64)
    logits = d["logits"].astype(np.float32)
    return utt_id, y_true, logits


def _save_npz(path: str, utt_id, y_true, logits):
    """Save (utt_id, y_true, logits) to compressed npz."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, utt_id=np.array(utt_id), y_true=y_true, logits=logits)


def _save_npz_conf(path: str, utt_id, y_true, conf_raw, conf_cal, correct01):
    """Save confidence calibration packs (for reliability diagram mode=\"confidence\")."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        utt_id=np.array(utt_id),
        y_true=y_true,
        conf_raw=np.asarray(conf_raw, dtype=np.float32),
        conf_cal=np.asarray(conf_cal, dtype=np.float32),
        correct01=np.asarray(correct01, dtype=np.float32),
    )


def calibrate_logits(
    val_logits: np.ndarray,
    val_y: np.ndarray,
    test_logits: np.ndarray,
    test_y: np.ndarray,
    num_classes: int,
    method: str = "temperature",
    n_bins: int = 15,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, object]:
    """
    Calibrate using validation logits, then evaluate on val/test.

    method:
        - "temperature": one scalar T
        - "vector": per-class scale+bias (more flexible)

    Returns:
        report_dict, calibrated_val_logits, calibrated_test_logits, calibrator_object
    """
    # Metrics before calibration
    pre_val_cls = summarize_cls(val_y, val_logits, num_classes)
    pre_tst_cls = summarize_cls(test_y, test_logits, num_classes)
    pre_val_cal = summarize_calibration(val_logits, val_y, num_classes, n_bins=n_bins)
    pre_tst_cal = summarize_calibration(test_logits, test_y, num_classes, n_bins=n_bins)

    # Fit calibrator on validation set (ONLY)
    if method == "temperature":
        calib = TemperatureScaling()
    elif method == "vector":
        calib = VectorScaling(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown method: {method}")

    calib.fit(val_logits, val_y)

    # Apply to val/test
    val_logits_c = calib.transform_logits(val_logits)
    test_logits_c = calib.transform_logits(test_logits)

    post_val_cls = summarize_cls(val_y, val_logits_c, num_classes)
    post_tst_cls = summarize_cls(test_y, test_logits_c, num_classes)
    post_val_cal = summarize_calibration(val_logits_c, val_y, num_classes, n_bins=n_bins)
    post_tst_cal = summarize_calibration(test_logits_c, test_y, num_classes, n_bins=n_bins)

    report = {
        "method": method,
        "pre": {"val": {"cls": pre_val_cls, "cal": pre_val_cal}, "test": {"cls": pre_tst_cls, "cal": pre_tst_cal}},
        "post": {"val": {"cls": post_val_cls, "cal": post_val_cal}, "test": {"cls": post_tst_cls, "cal": post_tst_cal}},
        "calibrator_state": calib.state_dict(),
    }
    return report, val_logits_c, test_logits_c, calib


def calibrate_quality(
    val_logits: np.ndarray,
    val_y: np.ndarray,
    test_logits: np.ndarray,
    test_y: np.ndarray,
    method: str = "conf_isotonic",
    n_bins: int = 15,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, np.ndarray], object]:
    """校准“质量/置信度”：把 max-softmax confidence 映射为 P(correct)。

    这不是对 logits 做 scaling（不会改变分类 argmax），而是做：
        conf_raw = max(softmax(logits))
        conf_cal = g(conf_raw)  ≈  P(y_pred == y_true | conf_raw)

    典型用途：Router 的“质量/可靠性”元特征（MoCaE/Quality estimation 风格）。

    返回:
      report: 包含 val/test 的 pre/post quality 指标（q_nll/q_ece/q_brier）
      val_pack/test_pack: 可直接写成 npz，用于画 reliability diagram（mode="confidence"）
      calibrator: 拟合后的 calibrator 对象
    """
    if method == "conf_isotonic":
        calib = ConfidenceIsotonic()
    elif method == "conf_linear":
        calib = ConfidenceLinear()
    else:
        raise KeyError(f"Unknown confidence calibration method: {method}")

    calib.fit(val_logits, val_y)

    def _pack(logits: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        # raw confidence + correctness
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z) / np.clip(np.exp(z).sum(axis=1, keepdims=True), 1e-12, None)
        conf_raw = p.max(axis=1).astype(np.float32)
        pred = p.argmax(axis=1)
        correct01 = (pred == y).astype(np.float32)
        conf_cal = calib.transform_confidence(conf_raw).astype(np.float32)
        return {
            "conf_raw": conf_raw,
            "conf_cal": conf_cal,
            "correct01": correct01,
        }

    val_pack = _pack(val_logits, val_y)
    test_pack = _pack(test_logits, test_y)

    report = {
        "mode": "quality",
        "method": method,
        "val_pre": summarize_quality(val_pack["conf_raw"], val_pack["correct01"], n_bins=n_bins),
        "val_post": summarize_quality(val_pack["conf_cal"], val_pack["correct01"], n_bins=n_bins),
        "test_pre": summarize_quality(test_pack["conf_raw"], test_pack["correct01"], n_bins=n_bins),
        "test_post": summarize_quality(test_pack["conf_cal"], test_pack["correct01"], n_bins=n_bins),
    }
    return report, val_pack, test_pack, calib


def calibrate_quality_run_dir(
    run_dir: str,
    method: str = "conf_isotonic",
    n_bins: int = 15,
    in_val: str = "preds/val_logits.npz",
    in_test: str = "preds/test_logits.npz",
) -> Dict[str, Any]:
    """对某个 run_dir 做“质量/置信度校准”（不改 logits）。

    Writes:
      run_dir/calibration/conf_<method>.json
      run_dir/calibration/conf_report_<method>.json
      run_dir/calibration/{val,test}_conf_<method>.npz  （给 plot_reliability_diagram(mode="confidence") 用）
    """
    val_path = os.path.join(run_dir, in_val)
    test_path = os.path.join(run_dir, in_test)

    utt_v, y_v, log_v = _load_npz(val_path)
    utt_t, y_t, log_t = _load_npz(test_path)

    report, val_pack, test_pack, calib = calibrate_quality(
        val_logits=log_v, val_y=y_v,
        test_logits=log_t, test_y=y_t,
        method=method, n_bins=n_bins,
    )

    out_dir = os.path.join(run_dir, "calibration")
    os.makedirs(out_dir, exist_ok=True)

    # calibrator
    cal_path = os.path.join(out_dir, f"conf_{method}.json")
    save_calibrator(cal_path, calib)

    # reports + per-split packs
    rep_path = os.path.join(out_dir, f"conf_report_{method}.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _save_npz_conf(os.path.join(out_dir, f"val_conf_{method}.npz"), utt_v, y_v, val_pack["conf_raw"], val_pack["conf_cal"], val_pack["correct01"])
    _save_npz_conf(os.path.join(out_dir, f"test_conf_{method}.npz"), utt_t, y_t, test_pack["conf_raw"], test_pack["conf_cal"], test_pack["correct01"])

    return report

def calibrate_run_dir(
    run_dir: str,
    num_classes: int,
    method: str = "temperature",
    n_bins: int = 15,
    in_val: str = "preds/val_logits.npz",
    in_test: str = "preds/test_logits.npz",
) -> Dict[str, Any]:
    """
    Calibrate a completed training run directory (runs/<exp>/fold_k).

    Writes artifacts under:
        run_dir/calibration/
        run_dir/preds/*_logits_calib.npz
    """
    val_path = os.path.join(run_dir, in_val)
    test_path = os.path.join(run_dir, in_test)

    utt_v, y_v, log_v = _load_npz(val_path)
    utt_t, y_t, log_t = _load_npz(test_path)

    report, log_v_c, log_t_c, calib = calibrate_logits(
        val_logits=log_v, val_y=y_v,
        test_logits=log_t, test_y=y_t,
        num_classes=num_classes,
        method=method,
        n_bins=n_bins,
    )

    out_dir = os.path.join(run_dir, "calibration")
    os.makedirs(out_dir, exist_ok=True)

    # Save report + calibrator parameters
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    save_calibrator(os.path.join(out_dir, f"{method}.json"), calib)

    # Save calibrated logits (for router/fusion)
    _save_npz(os.path.join(run_dir, "preds", "val_logits_calib.npz"), utt_v, y_v, log_v_c)
    _save_npz(os.path.join(run_dir, "preds", "test_logits_calib.npz"), utt_t, y_t, log_t_c)

    return report


# CLI remains available, but not required in PyCharm.
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--method", choices=["temperature", "vector"], default="temperature")
    ap.add_argument("--n_bins", type=int, default=15)
    args = ap.parse_args()

    rep = calibrate_run_dir(args.run_dir, num_classes=args.num_classes, method=args.method, n_bins=args.n_bins)
    print("Saved calibration report:", os.path.join(args.run_dir, "calibration", "report.json"))
    print("Test post ECE:", rep["post"]["test"]["cal"]["ece"], "Brier:", rep["post"]["test"]["cal"]["brier"])


if __name__ == "__main__":
    main()


## Reliability diagram plotting (可选：画校准曲线)
# 说明：
# - 这个函数原来在 scripts/plot_reliability.py，为了减少文件数量并方便“一键运行”，合并到本文件。
# - 你可以在校准后调用它，把 reliability diagram 保存成 png。

def plot_reliability_diagram(npz_path: str, out_png: str, mode: str = "logits", n_bins: int = 15) -> str:
    """从 npz（val/test logits 或 confidence）画 reliability diagram 并保存为 PNG。

    参数:
      npz_path: 例如 runs/<exp>/fold_0/preds/test_logits.npz（或你保存的 conf_*.npz）
      out_png:  输出路径
      mode:
        - "logits": npz 里包含 logits + y_true
        - "confidence": npz 里包含 conf_raw/conf_cal + correct01
    返回:
      out_png
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from src.ser.calibration import (
        reliability_curve_from_logits,
        reliability_curve_from_confidence,
    )

    d = np.load(npz_path, allow_pickle=True)

    if mode == "logits":
        y_true = d["y_true"].astype(np.int64)
        logits = d["logits"].astype(np.float32)
        c, acc, conf = reliability_curve_from_logits(logits, y_true, n_bins=n_bins)
        title = "Reliability Diagram (softmax logits)"
        x_axis = conf
    else:
        conf = d["conf_cal"].astype(np.float32) if "conf_cal" in d else d["conf_raw"].astype(np.float32)
        correct01 = d["correct01"].astype(np.float32)
        c, acc, conf = reliability_curve_from_confidence(conf, correct01, n_bins=n_bins)
        title = "Reliability Diagram (max-prob confidence)"
        x_axis = c

    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.plot(x_axis, acc, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    return out_png
