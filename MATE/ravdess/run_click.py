from __future__ import annotations

import os
import sys
from typing import List, Optional

from src.ser.config import load_yaml
from scripts.train import train_from_yaml
from scripts.calibrate import calibrate_run_dir, plot_reliability_diagram


try:
    from scripts.calibrate import calibrate_quality_run_dir
except Exception:
    calibrate_quality_run_dir = None

try:
    from scripts.analyze_expert import analyze_after_train
except Exception:
    analyze_after_train = None


CONFIG = {
    "dataset_yaml": "configs/datasets/ravdess.yaml",
    "expert_yamls": [

        # "audio_whisper_cnn_attention.yaml",
        #"audio_whisper_cnn_attentive.yaml",
        # "audio_whisper_transformer_attentive.yaml",

        # "configs/experts/fusion_av_pool_mlp.yaml",
        # "fusion_wavlm_hsemotion_xattn.yaml",
        # "fusion__whisper_hsemotion_multlite.yaml",
        # "fusion_whisper_hsemotion_lmf.yaml",
        # "fusion_whisper_film.yaml",
        # "fusion_whisper_TFN.yaml",
        # "configs/experts/fusion_av_gated.yaml",

        #"audio_wavlm_transformer.yaml",
        #"audio_wavlm_lstm.yaml",
        #"audio_wavlm_transformer.yaml",

        #"video_hsemotion_BiGRU.yaml",
        "video_hsemotion_asp.yaml",
        #"video_hsemotion_tcn.yaml",
        #"video_hsemotion.yaml",

        #"video_openface_tcn_gn.yaml",
        #"video_OpenFace-linear_attn.yaml",
        #"video_openface-transformer.yaml",


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
        #"fusion_wavlm_avg_all_hsemotion_xattn.yaml"
        #"fusion_wavlmall_film.yaml",
        #"fusion_wavlmall_hsemotion_lmf.yaml",
        #"fusion_wavlmall_TFN.yaml",
        #"configs/experts/fusion_av_pool_mlp_meta.yaml",



    ],
    "device": "cuda",
    "folds": None,  # None -> infer from dataset protocol

    "calib_methods": ["temperature", "vector"],
    "reliability_bins": 15,

    "plot_reliability": True,
    "plot_split": "test",

    "run_analysis": True,
    "analysis_splits": ["val", "test"],
    "analysis_mc_samples": 0,
    "analysis_save_embeddings": True,

    "run_quality_calibration": False,
    "quality_methods": ["conf_isotonic"],

    "skip_done": True,
}


# =========================


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _infer_folds(dataset_yaml: str) -> List[int]:
    ds_cfg = load_yaml(dataset_yaml)
    protocol = ds_cfg.get("protocol", {})
    ptype = str(protocol.get("type", "group_kfold"))
    if ptype == "repeated_holdout":
        repeats = int(protocol.get("repeats", 5))
        return list(range(repeats))
    n_splits = int(protocol.get("n_splits", 5))
    return list(range(n_splits))


def _num_classes(dataset_yaml: str) -> int:
    ds_cfg = load_yaml(dataset_yaml)
    return len(ds_cfg["labels"])


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
    _safe_rename(os.path.join(pred_dir, "val_logits_calib.npz"),
                 os.path.join(pred_dir, f"val_logits_calib_{method}.npz"))
    _safe_rename(os.path.join(pred_dir, "test_logits_calib.npz"),
                 os.path.join(pred_dir, f"test_logits_calib_{method}.npz"))


def _plot_reliability(run_dir: str, split: str, n_bins: int, calib_methods: List[str]) -> None:
    cal_dir = os.path.join(run_dir, "calibration")
    pred_dir = os.path.join(run_dir, "preds")
    os.makedirs(cal_dir, exist_ok=True)

    raw_npz = os.path.join(pred_dir, f"{split}_logits.npz")
    if os.path.exists(raw_npz):
        plot_reliability_diagram(raw_npz, os.path.join(cal_dir, f"reliability_{split}_raw.png"), mode="logits",
                                 n_bins=n_bins)

    for m in calib_methods:
        cal_npz = os.path.join(pred_dir, f"{split}_logits_calib_{m}.npz")
        if os.path.exists(cal_npz):
            plot_reliability_diagram(cal_npz, os.path.join(cal_dir, f"reliability_{split}_{m}.png"), mode="logits",
                                     n_bins=n_bins)


def run_all(
        dataset_yaml: str,
        expert_yamls: List[str],
        device: Optional[str],
        folds: Optional[List[int]],
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
):
    os.chdir(_repo_root())
    if folds is None:
        folds = _infer_folds(dataset_yaml)
    num_classes = _num_classes(dataset_yaml)

    for expert_yaml in expert_yamls:
        exp_cfg = load_yaml(expert_yaml)
        exp_name = exp_cfg.get("exp_name", os.path.splitext(os.path.basename(expert_yaml))[0])

        print("\n" + "=" * 100)
        print(f"[EXPERT] {exp_name}")
        print("yaml:", expert_yaml)
        print("folds:", folds)
        print("=" * 100)

        for fold in folds:
            run_dir = os.path.join("runs_2.0", exp_name, f"fold_{fold}")

            print("\n" + "-" * 80)
            print(f"[{exp_name}] Fold {fold}: TRAIN/TEST")
            print("-" * 80)

            if skip_done and _maybe_done(run_dir):
                print("[SKIP] already done:", run_dir)
            else:
                out = train_from_yaml(expert_yaml, dataset_yaml, fold=int(fold), device=device)
                run_dir = out["run_dir"]
                print("Test metrics:", out["metrics"]["test"])

            # logit calibration
            for m in calib_methods:
                print(f"\n[{exp_name}] Fold {fold}: CALIBRATE logits -> {m}")
                calibrate_run_dir(run_dir, num_classes=num_classes, method=m, n_bins=int(n_bins))
                _stash_calibration(run_dir, method=m)
                print("Saved:", os.path.join(run_dir, "calibration", f"report_{m}.json"))

            # optional quality calibration
            if run_quality:
                if calibrate_quality_run_dir is None:
                    print(
                        "[WARN] quality calibration not available (scripts/calibrate.py missing calibrate_quality_run_dir).")
                else:
                    for qm in quality_methods:
                        print(f"\n[{exp_name}] Fold {fold}: QUALITY CALIB -> {qm}")
                        calibrate_quality_run_dir(run_dir, method=qm, n_bins=int(n_bins))

            # optional analysis
            if run_analysis:
                if analyze_after_train is None:
                    print("[WARN] analysis not available (scripts/analyze_expert.py missing analyze_after_train).")
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
    ap = argparse.ArgumentParser("run_click_multi")
    ap.add_argument("--dataset", default=CONFIG["dataset_yaml"])
    ap.add_argument("--experts", nargs="+", required=True)
    ap.add_argument("--device", default=CONFIG["device"])
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--calib_methods", nargs="+", default=CONFIG["calib_methods"])
    ap.add_argument("--n_bins", type=int, default=CONFIG["reliability_bins"])
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--analysis", action="store_true")
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--plot_split", default=CONFIG["plot_split"], choices=["val", "test"])
    return ap.parse_args()


def main():
    # CLI
    if len(sys.argv) > 1:
        args = _parse_cli()
        run_all(
            dataset_yaml=args.dataset,
            expert_yamls=args.experts,
            device=args.device,
            folds=args.folds,
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
        return

    # PyCharm click-run (no args)
    if not CONFIG["expert_yamls"]:
        raise RuntimeError("CONFIG['expert_yamls'] is empty. Add at least one expert yaml.")
    run_all(
        dataset_yaml=CONFIG["dataset_yaml"],
        expert_yamls=CONFIG["expert_yamls"],
        device=CONFIG["device"],
        folds=CONFIG["folds"],
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


if __name__ == "__main__":
    main()
