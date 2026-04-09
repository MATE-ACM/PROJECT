from __future__ import annotations

"""
【文件作用】训练入口：按 dataset 配置划分 train/val/test，训练指定 expert，并保存 checkpoint 与 val/test logits。

Training entrypoint for a single expert (audio / video / fusion).

You can still run it from terminal, BUT it is designed to be called directly from Python
(e.g. in PyCharm / notebooks) via `train_from_yaml(...)`.

Key design:
- Experts are decoupled: expert.type determines which dataloader is used.
- Splits are speaker-disjoint GroupKFold (for CREMA-D) by default (see dataset config).
- Each fold produces:
    runs/<exp_name>/fold_k/checkpoints/best.pt
    runs/<exp_name>/fold_k/preds/{val,test}_logits(.npz)

Typical PyCharm usage:
    from scripts.train import train_from_yaml
    train_from_yaml("audio_ssl.yaml", "configs/datasets/cremad.yaml", fold=0)
"""

import os
import json
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.ser.config import load_yaml, deep_update
from src.ser.data.manifest import read_jsonl
from src.ser.data.splits import make_group_kfold_splits
from src.ser.data.dataloaders import make_loaders
from src.ser.experts import build_expert
from src.ser.losses import build_loss
from src.ser.metrics import summarize
from src.ser.train_utils import make_optimizer, make_scheduler


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(items: List[Dict[str, Any]], label2id: Dict[str, int], num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a list of items (train set only).

    This is used when loss.type is weighted_ce / focal / label_smoothing.
    """
    counts = np.zeros((num_classes,), dtype=np.int64)
    for it in items:
        counts[label2id[it["label"]]] += 1

    w = 1.0 / np.maximum(counts, 1)         # inverse frequency
    w = w * (num_classes / w.sum())         # normalize to mean ~1
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def eval_and_dump(
    model: torch.nn.Module,
    loader,
    device: str,
    num_classes: int,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run evaluation, returning metrics and optionally dumping logits to .npz for later calibration/fusion/router.
    """
    model.eval()
    all_logits, all_y, all_utt = [], [], []

    for batch in loader:
        # keep y on CPU for numpy ops
        y = batch["y"].numpy()

        # move inputs to GPU/CPU device
        for k in list(batch.keys()):
            if k.startswith("x_") or k == "y":
                batch[k] = batch[k].to(device)

        logits = model(batch).detach().cpu().numpy()

        all_logits.append(logits)
        all_y.append(y)
        all_utt += batch["utt_id"]

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)

    # Dump logits for calibrate.py and later router/fusion scripts
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, utt_id=np.array(all_utt), y_true=y_true, logits=logits)

    return summarize(y_true, logits, num_classes)


def _split_train_val_by_speaker(tr_items: List[Dict[str, Any]], group_key: str, seed: int, fold: int, val_ratio: float = 0.1):
    """
    Make a simple speaker-disjoint validation split inside the training set.

    NOTE: intentionally simple; you can swap it later to a strict nested-CV if needed.
    """
    speakers = sorted(list({it[group_key] for it in tr_items}))
    rng = np.random.RandomState(seed + 1000 + fold)
    rng.shuffle(speakers)
    n_val = max(1, int(val_ratio * len(speakers)))
    val_speakers = set(speakers[:n_val])

    va_items = [it for it in tr_items if it[group_key] in val_speakers]
    tr_items2 = [it for it in tr_items if it[group_key] not in val_speakers]
    return tr_items2, va_items


def train_from_dict(
    expert_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    fold: int = 0,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main callable API: train one expert for one fold.

    Returns:
        dict with paths + final val/test metrics.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Merge dataset config + expert config into one config dict
    cfg = deep_update(dataset_cfg, expert_cfg)

    # Label space
    labels = cfg["labels"]
    label2id = {l: i for i, l in enumerate(labels)}
    num_classes = len(labels)

    # Load manifest items
    items = read_jsonl(cfg["manifest_path"])

    protocol = cfg["protocol"]
    ptype = str(protocol.get("type", "group_kfold"))

    if ptype == "repeated_holdout":
        from src.ser.data.splits import make_repeated_holdout_splits

        folds, shuffled = make_repeated_holdout_splits(
            items,
            group_key=protocol["group_key"],
            repeats=int(protocol.get("repeats", 5)),
            seed=int(protocol.get("seed", 1337)),
            ratio_train=float(protocol.get("ratio_train", 0.70)),
            ratio_val=float(protocol.get("ratio_val", 0.15)),
            ratio_test=float(protocol.get("ratio_test", 0.15)),
        )

        fold_def = folds[fold]
        tr_items = [shuffled[i] for i in fold_def["train_idx"]]
        va_items = [shuffled[i] for i in fold_def["val_idx"]]
        te_items = [shuffled[i] for i in fold_def["test_idx"]]

    else:
        # default: group_kfold
        folds, shuffled = make_group_kfold_splits(
            items,
            group_key=protocol["group_key"],
            n_splits=int(protocol["n_splits"]),
            seed=int(protocol["seed"]),
        )

        fold_def = folds[fold]
        te_idx = set(fold_def["test_idx"])
        te_items = [shuffled[i] for i in fold_def["test_idx"]]
        tr_items = [shuffled[i] for i in fold_def["train_idx"] if i not in te_idx]

        # Train/Val split inside training (speaker-disjoint)
        val_ratio = float(protocol.get("val_ratio_in_train", 0.1))
        tr_items, va_items = _split_train_val_by_speaker(
            tr_items, group_key=protocol["group_key"], seed=int(protocol["seed"]), fold=fold, val_ratio=val_ratio
        )

    # Run directory
    exp_name = cfg.get("exp_name", "exp")
    run_dir = os.path.join("runs", exp_name, f"fold_{fold}")
    os.makedirs(run_dir, exist_ok=True)


    # 保存“本次训练真正使用的划分”（强烈推荐）
    # 后续 analyze_expert / router / fusion 都优先读这个，避免“脚本重建 split 导致对不齐/混数据”。
    splits = {
        "fold_id": int(fold),
        "train_utt_id": [it["utt_id"] for it in tr_items],
        "val_utt_id": [it["utt_id"] for it in va_items],
        "test_utt_id": [it["utt_id"] for it in te_items],
    }
    with open(os.path.join(run_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)


    # Reproducibility
    set_seed(int(protocol["seed"]) + 10 + fold)

    # Build model
    model = build_expert(cfg["expert"]).to(device)

    # Loss
    class_w = compute_class_weights(tr_items, label2id, num_classes).to(device)
    loss_cfg = cfg.get("loss", {"type": "ce"})
    use_w = loss_cfg.get("type") in ["weighted_ce", "focal", "label_smoothing"]
    loss_fn = build_loss(loss_cfg, class_w if use_w else None)

    # Dataloaders: selected automatically by expert.type prefix (audio_/video_/fusion)
    train_cfg = cfg["train"]
    tr_loader, va_loader, te_loader = make_loaders(
        tr_items, va_items, te_items,
        label2id, cfg, cfg["expert"]["type"],
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 4)),
    )

    # Optimizer & LR schedule
    opt = make_optimizer(model, float(train_cfg["lr"]), float(train_cfg.get("weight_decay", 0.0)))
    total_steps = int(train_cfg["epochs"]) * max(1, len(tr_loader))
    sch = make_scheduler(opt, train_cfg.get("scheduler", "cosine"), total_steps, float(train_cfg.get("warmup_ratio", 0.0)))

    # Checkpointing
    best_metric = -1e9
    best_path = os.path.join(run_dir, "checkpoints", "best.pt")
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    # Early stopping
    es = train_cfg.get("early_stop", {"enabled": False})
    patience = int(es.get("patience", 10))
    metric_name = es.get("metric", "uar")
    bad = 0

    grad_clip = float(train_cfg.get("grad_clip", 0.0))

    # Training loop
    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        pbar = tqdm(tr_loader, desc=f"[{exp_name} fold{fold}] epoch {epoch}")

        for batch in pbar:
            y = batch["y"].to(device)

            # move inputs
            for k in list(batch.keys()):
                if k.startswith("x_"):
                    batch[k] = batch[k].to(device)

            logits = model(batch)

            # build_loss may return nn.Module or callable focal loss
            loss = loss_fn(logits, y) if callable(loss_fn) else loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()
            sch.step()

            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

        # Evaluate on val
        val_out = eval_and_dump(model, va_loader, device, num_classes, save_path=None)
        cur = float(val_out.get(metric_name, val_out["uar"]))

        # Append a log line for epoch metrics
        with open(os.path.join(run_dir, "val_metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, **val_out}, ensure_ascii=False) + "\n")

        # Save best
        if cur > best_metric:
            best_metric = cur
            torch.save({"model": model.state_dict(), "cfg": cfg, "label2id": label2id, "fold_id": int(fold)}, best_path)
            bad = 0
        else:
            bad += 1

        if es.get("enabled", False) and bad >= patience:
            break

    # Load best model for final dumps
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    save_preds = cfg.get("log", {}).get("save_preds", True)
    val_pred_path = os.path.join(run_dir, "preds", "val_logits.npz") if save_preds else None
    te_pred_path = os.path.join(run_dir, "preds", "test_logits.npz") if save_preds else None

    val_out = eval_and_dump(model, va_loader, device, num_classes, save_path=val_pred_path)
    te_out = eval_and_dump(model, te_loader, device, num_classes, save_path=te_pred_path)

    # Write final metrics
    with open(os.path.join(run_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"val": val_out, "test": te_out}, f, ensure_ascii=False, indent=2)

    return {
        "run_dir": run_dir,
        "best_ckpt": best_path,
        "metrics": {"val": val_out, "test": te_out},
    }


def train_from_yaml(expert_yaml_path: str, dataset_yaml_path: str, fold: int = 0, device: Optional[str] = None) -> Dict[str, Any]:
    """Convenience wrapper for PyCharm usage: provide yaml paths, get back metrics and artifact paths."""
    expert_cfg = load_yaml(expert_yaml_path)
    dataset_cfg = load_yaml(dataset_yaml_path)
    return train_from_dict(expert_cfg, dataset_cfg, fold=fold, device=device)


# CLI remains available, but not required in PyCharm.
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="audio_npy_universal.yaml", help="expert config yaml")
    ap.add_argument("--dataset", default="configs/datasets/cremad.yaml", help="dataset config yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    out = train_from_yaml(args.config, args.dataset, fold=args.fold, device=args.device)
    print("Done:", out["metrics"]["test"])


if __name__ == "__main__":
    main()
