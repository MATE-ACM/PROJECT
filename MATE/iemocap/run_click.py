from __future__ import annotations

import os
import sys
import copy
import glob
from typing import List, Optional, Dict, Any, Tuple

from src.ser.config import load_yaml
from scripts.train import train_from_dict
from scripts.calibrate import calibrate_run_dir, plot_reliability_diagram

try:
    from scripts.calibrate import calibrate_quality_run_dir
except Exception:
    calibrate_quality_run_dir = None

try:
    from scripts.analyze_expert import analyze_after_train
except Exception:
    analyze_after_train = None


# =========================

# =========================
CONFIG = {
    "dataset_yaml": "configs/datasets/iemocap.yaml",

    # 4way / 6way / None
    "label_scheme": None,

    # 是否把 label_scheme 追加到 exp_name（防止 runs 被覆盖）
    "append_label_scheme_to_exp_name": True,

    "expert_yamls": [
        # ======== openface video ========
        #"audio_whisper_cnn_attention.yaml",

        #"audio_whisper_cnn_attentive.yaml",
        #"audio_whisper_transformer_attentive.yaml",
        #"fusion__whisper_hsemotion_multlite.yaml",
        #"fusion_av_gated.yaml",
        #"fusion_av_pool_mlp.yaml",
        #"fusion_av_whisper_xattn.yaml",
        #"fusion_whisper_film.yaml",
        #"fusion_whisper_hsemotion_lmf.yaml",
        #"fusion_whisper_TFN.yaml",


        #"video_hsemotion.yaml",

        #"video_hsemotion_asp.yaml",
       # "video_hsemotion_BiGRU.yaml",
        #"video_hsemotion_tcn.yaml",

        #"audio_whisper-txt_cnn_attention.yaml",


        #"audio_wavlm_avg_last12_cnn.yaml",
        #"audio_wavlm_avg_last12_lstm.yaml",
        #"audio_wavlm_avg_last12_transformer.yaml",
        #"fusion__wavlm12_hsemotion_multlite.yaml",
        #"fusion_av_gated_wavlm_avg_last12.yaml",
        #"fusion_av_pool_mlp_wavlm_avg_last12.yaml",
        #"fusion_wavlm12_film.yaml",
        #"fusion_wavlm12_hsemotion_lmf.yaml",
        #"fusion_wavlm12_TFN.yaml",
        #"fusion_wavlm_avg_last12_hsemotion_xattn.yaml",


        #"audio_wavlm_avg_all_cnn.yaml",
        #"audio_wavlm_avg_all_lstm.yaml",
        #"audio_wavlm_avg_all_transformer.yaml",
        #"fusion__wavlmall_hsemotion_multlite.yaml",
        #"fusion_av_gated_wavlm_avg_all.yaml",
        #"fusion_av_pool_mlp_wavlm_avg_all.yaml",
        #"fusion_wavlm_avg_all_hsemotion_xattn.yaml",
        #"fusion_wavlmall_film.yaml",
        #"fusion_wavlmall_hsemotion_lmf.yaml",
        #"fusion_wavlmall_TFN.yaml",



        #"configs/experts/txt/txt_roberta_biGRU_asp.yaml",
        #"configs/experts/txt/txt_roberta_tcn_gn_asp.yaml",
        #"configs/experts/txt/txt_roberta_transformer_asp.yaml",
        #"configs/experts/txt/txt_roberta_utt_mlp.yaml",


        "wavlm_txt_lmf.yaml",
        "whisper_txt_lmf.yaml",

        "iemocap_avt_wavlm_hsemotion_txt_pool_mlp.yaml",
        "iemocap_avt_wavlm_hsemotion_txt_xattn.yaml",


        "iemocap_avt_whisper_hsemotion_txt_pool_mlp.yaml",
        "iemocap_avt_whisper_hsemotion_txt_xattn.yaml",

        "txt_hsemotion_pool_mlp.yaml",




        #"video_OpenFace-linear_attn.yaml",
        #"video_openface-transformer.yaml",
        #"video_openface_tcn_gn.yaml",
    ],

    "device": "cuda",
    "folds": None,  # None -> infer from dataset protocol

    # ------- calibration / reliability -------
    "calib_methods": ["temperature", "vector"],
    "reliability_bins": 15,
    "plot_reliability": True,
    "plot_split": "test",  # "val" or "test"

    # ------- analysis -------
    "run_analysis": True,
    "analysis_splits": ["val", "test"],
    "analysis_mc_samples": 0,
    "analysis_save_embeddings": True,

    # ------- optional quality calibration -------
    "run_quality_calibration": False,
    "quality_methods": ["conf_isotonic"],

    # skip if best.pt + test_logits + final_metrics exist
    "skip_done": True,
}
# =========================


def _repo_root() -> str:
 
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def _resolve_yaml_path(p: str) -> str:
    """If p doesn't exist, try to locate by basename under configs/."""
    p = str(p)
    if os.path.exists(p):
        return p

    bn = os.path.basename(p)#从完整的文件路径中提取文件名 将yaml单独提取出来
    hits = glob.glob(os.path.join("configs", "**", bn), recursive=True)#在路径下搜索yaml
    hits = [h for h in hits if os.path.isfile(h)]

    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise FileNotFoundError(
            f"YAML not found: {p}\nBut found multiple candidates:\n" + "\n".join(hits)
        )
    raise FileNotFoundError(f"YAML not found: {p}")


def _infer_folds_from_cfg(ds_cfg: Dict[str, Any]) -> List[int]:
    protocol = ds_cfg.get("protocol", {})
    ptype = str(protocol.get("type", "group_kfold"))
    if ptype == "repeated_holdout":
        repeats = int(protocol.get("repeats", 5))
        return list(range(repeats))
    n_splits = int(protocol.get("n_splits", 5))
    return list(range(n_splits))


def _maybe_done(run_dir: str) -> bool:
    return (
        os.path.exists(os.path.join(run_dir, "checkpoints", "best.pt"))
        and os.path.exists(os.path.join(run_dir, "preds", "test_logits.npz"))
        and os.path.exists(os.path.join(run_dir, "final_metrics.json"))
    )


def _safe_rename(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    if os.path.exists(src):
        os.rename(src, dst)


def _stash_calibration(run_dir: str, method: str) -> None:
    """
    calibrate_run_dir writes:
      calibration/report.json
      preds/val_logits_calib.npz
      preds/test_logits_calib.npz
    rename them to avoid overwrite across methods.
    """
    cal_dir = os.path.join(run_dir, "calibration")
    pred_dir = os.path.join(run_dir, "preds")
    _safe_rename(os.path.join(cal_dir, "report.json"), os.path.join(cal_dir, f"report_{method}.json"))
    _safe_rename(os.path.join(pred_dir, "val_logits_calib.npz"), os.path.join(pred_dir, f"val_logits_calib_{method}.npz"))
    _safe_rename(os.path.join(pred_dir, "test_logits_calib.npz"), os.path.join(pred_dir, f"test_logits_calib_{method}.npz"))


def _plot_reliability(run_dir: str, split: str, n_bins: int, calib_methods: List[str]) -> None:
    cal_dir = os.path.join(run_dir, "calibration")
    pred_dir = os.path.join(run_dir, "preds")
    os.makedirs(cal_dir, exist_ok=True)

    raw_npz = os.path.join(pred_dir, f"{split}_logits.npz")
    if os.path.exists(raw_npz):
        plot_reliability_diagram(
            raw_npz,
            os.path.join(cal_dir, f"reliability_{split}_raw.png"),
            mode="logits",
            n_bins=n_bins,
        )

    for m in calib_methods:
        cal_npz = os.path.join(pred_dir, f"{split}_logits_calib_{m}.npz")
        if os.path.exists(cal_npz):
            plot_reliability_diagram(
                cal_npz,
                os.path.join(cal_dir, f"reliability_{split}_{m}.png"),
                mode="logits",
                n_bins=n_bins,
            )


def _apply_label_scheme(ds_cfg: Dict[str, Any], override_scheme: Optional[str]) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Supports dataset yaml like:
      label_scheme: 6way
      schemes:
        4way: { manifest_path: ..., labels: [...] }
        6way: { manifest_path: ..., labels: [...] }
    """
    ds_cfg = copy.deepcopy(ds_cfg)
    scheme = override_scheme or ds_cfg.get("label_scheme", None)

    schemes = ds_cfg.get("schemes", None)
    if not scheme or not isinstance(schemes, dict):
        return ds_cfg, scheme

    if scheme not in schemes:
        raise KeyError(f"label_scheme='{scheme}' not found in dataset_cfg.schemes keys={list(schemes.keys())}")

    pick = schemes[scheme] or {}
    ds_cfg["label_scheme"] = scheme
    if "manifest_path" in pick:
        ds_cfg["manifest_path"] = pick["manifest_path"]
    if "labels" in pick:
        ds_cfg["labels"] = pick["labels"]
    if "label_map" in pick:
        ds_cfg["label_map"] = pick["label_map"]

    return ds_cfg, scheme


def _patch_expert_num_classes(exp_cfg: Dict[str, Any], num_classes: int) -> Dict[str, Any]:
    exp_cfg = copy.deepcopy(exp_cfg)
    if "expert" in exp_cfg and isinstance(exp_cfg["expert"], dict):
        exp_cfg["expert"]["num_classes"] = int(num_classes)
    else:
        exp_cfg["num_classes"] = int(num_classes)
    return exp_cfg


def _maybe_suffix_exp_name(exp_cfg: Dict[str, Any], scheme: Optional[str], enable: bool) -> Dict[str, Any]:
    if not enable or not scheme:
        return exp_cfg

    exp_cfg = copy.deepcopy(exp_cfg)
    exp_name = exp_cfg.get("exp_name", None)
    if not exp_name:
        return exp_cfg

    suf = f"_{scheme}"
    if not str(exp_name).endswith(suf):
        exp_cfg["exp_name"] = str(exp_name) + suf
    return exp_cfg


def run_all(
    dataset_yaml: str,
    expert_yamls: List[str],
    device: Optional[str],
    folds: Optional[List[int]],
    label_scheme: Optional[str],
    append_scheme_to_exp_name: bool,
    calib_methods: List[str],
    n_bins: int,
    plot_reliability: bool,
    plot_split: str,
    run_analysis: bool,
    analysis_splits: List[str],
    analysis_mc: int,
    analysis_save_embeddings: bool,
    run_quality: bool,
    quality_methods: List[str],
    skip_done: bool,
) -> None:
    os.chdir(_repo_root())

    dataset_yaml = _resolve_yaml_path(dataset_yaml)
    ds_cfg0 = load_yaml(dataset_yaml)
    ds_cfg, scheme = _apply_label_scheme(ds_cfg0, override_scheme=label_scheme)

    if folds is None:
        folds = _infer_folds_from_cfg(ds_cfg)

    labels = ds_cfg.get("labels", None)
    if not labels:
        raise KeyError("dataset config missing 'labels' after applying label_scheme.")
    num_classes = len(labels)

    print("=" * 100)
    print("[DATASET]", dataset_yaml)
    if scheme:
        print("label_scheme:", scheme)
    print("num_classes:", num_classes)
    print("folds:", folds)
    print("=" * 100)

    for expert_yaml in expert_yamls:
        expert_yaml = _resolve_yaml_path(expert_yaml)
        exp_cfg0 = load_yaml(expert_yaml)

        exp_cfg = _patch_expert_num_classes(exp_cfg0, num_classes=num_classes)
        exp_cfg = _maybe_suffix_exp_name(exp_cfg, scheme=scheme, enable=append_scheme_to_exp_name)

        exp_name = exp_cfg.get("exp_name", os.path.splitext(os.path.basename(expert_yaml))[0])

        print("\n" + "=" * 100)
        print(f"[EXPERT] {exp_name}")
        print("yaml:", expert_yaml)
        print("=" * 100)

        for fold in folds:
            expected_run_dir = os.path.join("runs", exp_name, f"fold_{fold}")

            print("\n" + "-" * 80)
            print(f"[{exp_name}] Fold {fold}: TRAIN/TEST")
            print("-" * 80)

            if skip_done and _maybe_done(expected_run_dir):
                run_dir = expected_run_dir
                print("[SKIP] already done:", run_dir)
            else:
                out = train_from_dict(exp_cfg, ds_cfg, fold=int(fold), device=device)
                run_dir = out["run_dir"]
                print("Test metrics:", out["metrics"]["test"])

            for m in calib_methods:
                print(f"\n[{exp_name}] Fold {fold}: CALIBRATE logits -> {m}")
                calibrate_run_dir(run_dir, num_classes=num_classes, method=m, n_bins=int(n_bins))
                _stash_calibration(run_dir, method=m)
                print("Saved:", os.path.join(run_dir, "calibration", f"report_{m}.json"))

            if run_quality:
                if calibrate_quality_run_dir is None:
                    print("[WARN] quality calibration not available (calibrate_quality_run_dir missing).")
                else:
                    for qm in quality_methods:
                        print(f"\n[{exp_name}] Fold {fold}: QUALITY CALIB -> {qm}")
                        calibrate_quality_run_dir(run_dir, method=qm, n_bins=int(n_bins))

            if run_analysis:
                if analyze_after_train is None:
                    print("[WARN] analysis not available (analyze_after_train missing).")
                else:
                    print(f"\n[{exp_name}] Fold {fold}: ANALYZE")
                    analyze_after_train(
                        run_dir=run_dir,
                        splits=analysis_splits,
                        mc_samples=int(analysis_mc),
                        save_embeddings=bool(analysis_save_embeddings),
                        device=device,
                    )

            if plot_reliability:
                try:
                    _plot_reliability(run_dir, split=plot_split, n_bins=int(n_bins), calib_methods=calib_methods)
                    print("Reliability plots:", os.path.join(run_dir, "calibration"))
                except Exception as e:
                    print("[WARN] plot reliability failed:", e)

    print("\nAll done.")


def _parse_cli():
    import argparse

    ap = argparse.ArgumentParser("run_iemocap_click_multi")
    ap.add_argument("--dataset", default=CONFIG["dataset_yaml"])
    ap.add_argument("--experts", nargs="+", default=None, help="expert yaml list (optional).")
    ap.add_argument("--device", default=CONFIG["device"])
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--label_scheme", default=None, help="Override label scheme: 4way/6way (optional).")
    ap.add_argument("--no_suffix_scheme", action="store_true", help="Disable appending _4way/_6way to exp_name.")
    ap.add_argument("--calib_methods", nargs="+", default=CONFIG["calib_methods"])
    ap.add_argument("--n_bins", type=int, default=CONFIG["reliability_bins"])
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--analysis", action="store_true")
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--plot_split", default=CONFIG["plot_split"], choices=["val", "test"])
    return ap.parse_args()


def main() -> int:
    # CLI mode
    if len(sys.argv) > 1:
        args = _parse_cli()
        expert_list = args.experts if args.experts is not None else CONFIG["expert_yamls"]
        if not expert_list:
            raise RuntimeError("No experts provided. Use --experts ... or set CONFIG['expert_yamls'].")

        run_all(
            dataset_yaml=args.dataset,
            expert_yamls=expert_list,
            device=args.device,
            folds=args.folds,
            label_scheme=args.label_scheme,
            append_scheme_to_exp_name=(not bool(args.no_suffix_scheme)) and bool(CONFIG["append_label_scheme_to_exp_name"]),
            calib_methods=args.calib_methods,
            n_bins=args.n_bins,
            plot_reliability=bool(args.plot),
            plot_split=args.plot_split,
            run_analysis=bool(args.analysis),
            analysis_splits=CONFIG["analysis_splits"],
            analysis_mc=CONFIG["analysis_mc_samples"],
            analysis_save_embeddings=CONFIG["analysis_save_embeddings"],
            run_quality=bool(args.quality),
            quality_methods=CONFIG["quality_methods"],
            skip_done=bool(args.skip_done),
        )
        return 0

    # Click-run mode (PyCharm etc.)
    if not CONFIG["expert_yamls"]:
        raise RuntimeError("CONFIG['expert_yamls'] is empty. Add at least one expert yaml.")

    run_all(
        dataset_yaml=CONFIG["dataset_yaml"],
        expert_yamls=CONFIG["expert_yamls"],
        device=CONFIG["device"],
        folds=CONFIG["folds"],
        label_scheme=CONFIG["label_scheme"],
        append_scheme_to_exp_name=bool(CONFIG["append_label_scheme_to_exp_name"]),
        calib_methods=CONFIG["calib_methods"],
        n_bins=CONFIG["reliability_bins"],
        plot_reliability=CONFIG["plot_reliability"],
        plot_split=CONFIG["plot_split"],
        run_analysis=CONFIG["run_analysis"],
        analysis_splits=CONFIG["analysis_splits"],
        analysis_mc=CONFIG["analysis_mc_samples"],
        analysis_save_embeddings=CONFIG["analysis_save_embeddings"],
        run_quality=CONFIG["run_quality_calibration"],
        quality_methods=CONFIG["quality_methods"],
        skip_done=CONFIG["skip_done"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
