from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Optional, Tuple

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
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _infer_numeric_label_mode(items: List[Dict[str, Any]], num_classes: int, scan_n: int = 2000) -> Tuple[bool, int]:
    """
    检测 manifest 里 label 是否是数字（int 或 '6' 这种 digit str）。
    返回：(is_numeric, offset)
      - is_numeric=False：label 基本是字符串类别，不做处理
      - is_numeric=True：
          offset=0  表示 0-based (0..C-1)
          offset=-1 表示 1-based (1..C) -> 转为 0-based 需要减 1
    只在非常明确的情况下才自动判断 1-based。
    """
    vals: List[int] = []
    for it in items[: min(len(items), scan_n)]:
        lb = it.get("label", None)
        if isinstance(lb, (int, np.integer)):
            vals.append(int(lb))
            continue
        if isinstance(lb, str):
            s = lb.strip()
            if s.isdigit():
                vals.append(int(s))

    if not vals:
        return False, 0

    mn, mx = int(min(vals)), int(max(vals))

    # 典型 1..C
    if mn == 1 and mx == num_classes:
        print(f"[WARN] manifest labels look 1-based ({mn}..{mx}). Will convert to 0-based by subtracting 1.")
        return True, -1

    # 典型 0..C-1
    if mn == 0 and mx == num_classes - 1:
        return True, 0

    # 其它情况：不擅自猜
    print(f"[WARN] manifest has numeric labels but range looks unusual: min={mn}, max={mx}, num_classes={num_classes}. "
          f"Will treat as 0-based only for 0..C-1; other ids will raise.")
    return True, 0


def _canonical_label_scheme(x: Any) -> Optional[str]:
    """Normalize scheme names like 4/4way/6/6way.

    We keep this intentionally permissive so you can write:
      - 4 / "4" / "4way" / "4-way"
      - 6 / "6" / "6way" / "6-way"
    """
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"4", "4way", "4-way", "four", "4cls", "4class", "4-class"}:
        return "4way"
    if s in {"6", "6way", "6-way", "six", "6cls", "6class", "6-class"}:
        return "6way"
    return s


def _apply_label_scheme_inplace(cfg: Dict[str, Any]) -> None:
    """Apply cfg.label_scheme + cfg.schemes to override manifest_path / labels.

    This is the "统一开关" so you only change one place.
    """
    schemes = cfg.get("schemes", None)
    if not isinstance(schemes, dict) or not schemes:
        return

    scheme = _canonical_label_scheme(cfg.get("label_scheme") or cfg.get("scheme"))
    if not scheme:
        return

    if scheme not in schemes:
        raise KeyError(f"Unknown label_scheme='{scheme}'. Available: {list(schemes.keys())}")

    sc = schemes.get(scheme) or {}
    if not isinstance(sc, dict):
        raise TypeError(f"schemes['{scheme}'] must be a dict, got: {type(sc)}")

    # override top-level fields used by training
    if "manifest_path" in sc and sc["manifest_path"] is not None:
        cfg["manifest_path"] = sc["manifest_path"]
    if "labels" in sc and sc["labels"] is not None:
        cfg["labels"] = sc["labels"]

    # record the effective scheme
    cfg["label_scheme"] = scheme


def _build_label2id(labels: List[Any], num_classes: int, numeric: bool, offset: int) -> Dict[Any, int]:
    """
    生成一个“更鲁棒”的 label2id：
    - 始终包含 config 里的字符串标签映射（neutral->0 等）
    - 如果 manifest 是数字标签，则额外支持：
        offset=0 :  '0'..'C-1' / 0..C-1
        offset=-1:  '1'..'C'   / 1..C    映射到 0..C-1
    """
    label2id: Dict[Any, int] = {}
    for i, lb in enumerate(labels):
        # dataset 里会 str(it["label"])，所以 key 用 str(lb) 一定要有
        label2id[str(lb)] = i
        label2id[lb] = i

    if numeric:
        if offset == -1:
            # 1..C -> 0..C-1
            for k in range(1, num_classes + 1):
                label2id[k] = k - 1
                label2id[str(k)] = k - 1
        else:
            # 0..C-1
            for k in range(0, num_classes):
                label2id[k] = k
                label2id[str(k)] = k

    return label2id


def _label_to_id(lb: Any, label2id: Dict[Any, int], num_classes: int) -> int:
    """
    把 manifest 的 label（可能是 'angry' / 6 / '6'）转成 class id。
    """
    if lb in label2id:
        return int(label2id[lb])

    s = str(lb).strip()
    if s in label2id:
        return int(label2id[s])

    # 兜底：如果是数值但没被 build_label2id 接纳，就明确报错
    try:
        v = int(s)
        if 0 <= v < num_classes:
            return v
    except Exception:
        pass

    raise KeyError(f"Unknown label={lb} (as str='{s}'). label2id keys sample={list(label2id.keys())[:20]}...")


def compute_class_weights(items: List[Dict[str, Any]], label2id: Dict[Any, int], num_classes: int) -> torch.Tensor:
    counts = np.zeros((num_classes,), dtype=np.int64)
    for it in items:
        y = _label_to_id(it["label"], label2id, num_classes)
        counts[y] += 1

    w = 1.0 / np.maximum(counts, 1)
    w = w * (num_classes / w.sum())
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def eval_and_dump(
    model: torch.nn.Module,
    loader,
    device: str,
    num_classes: int,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    model.eval()
    all_logits, all_y, all_utt = [], [], []

    for batch in loader:
        y = batch["y"].numpy()

        for k in list(batch.keys()):
            if k.startswith("x_") or k == "y":
                batch[k] = batch[k].to(device)

        logits = model(batch).detach().cpu().numpy()
        all_logits.append(logits)
        all_y.append(y)
        all_utt += batch["utt_id"]

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, utt_id=np.array(all_utt), y_true=y_true, logits=logits)

    return summarize(y_true, logits, num_classes)


def _split_train_val_by_speaker(
    tr_items: List[Dict[str, Any]],
    group_key: str,
    seed: int,
    fold: int,
    val_ratio: float = 0.1
):
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
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = deep_update(dataset_cfg, expert_cfg)

    # ---- unified switch: label_scheme (4way/6way) ----
    _apply_label_scheme_inplace(cfg)

    labels = cfg["labels"]
    num_classes = len(labels)

    # Make sure the expert head matches the dataset (do NOT rely on each expert yaml)
    if "expert" not in cfg or not isinstance(cfg["expert"], dict):
        raise KeyError("Missing 'expert' config section.")
    cfg["expert"]["num_classes"] = int(num_classes)

    items = read_jsonl(cfg["manifest_path"])

    numeric, offset = _infer_numeric_label_mode(items, num_classes=num_classes)
    label2id = _build_label2id(labels, num_classes=num_classes, numeric=numeric, offset=offset)

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
        # group_kfold (+ optional blocks)
        kfold_method = str(protocol.get("kfold_method", protocol.get("kfold", "sklearn")))

    
        try:
            folds, shuffled = make_group_kfold_splits(
                items,
                group_key=protocol["group_key"],
                n_splits=int(protocol["n_splits"]),
                seed=int(protocol["seed"]),
                method=kfold_method,   # 新版才支持
            )
        except TypeError:
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

        val_ratio = float(protocol.get("val_ratio_in_train", 0.1))
        tr_items, va_items = _split_train_val_by_speaker(
            tr_items, group_key=protocol["group_key"], seed=int(protocol["seed"]), fold=fold, val_ratio=val_ratio
        )

    exp_name = cfg.get("exp_name", "exp")
    run_dir = os.path.join("runs", exp_name, f"fold_{fold}")
    os.makedirs(run_dir, exist_ok=True)

    splits = {
        "fold_id": int(fold),
        "train_utt_id": [it["utt_id"] for it in tr_items],
        "val_utt_id": [it["utt_id"] for it in va_items],
        "test_utt_id": [it["utt_id"] for it in te_items],
    }
    with open(os.path.join(run_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    set_seed(int(protocol["seed"]) + 10 + fold)

    model = build_expert(cfg["expert"]).to(device)

    class_w = compute_class_weights(tr_items, label2id, num_classes).to(device)
    loss_cfg = cfg.get("loss", {"type": "ce"})
    use_w = loss_cfg.get("type") in ["weighted_ce", "focal", "label_smoothing"]
    loss_fn = build_loss(loss_cfg, class_w if use_w else None)

    train_cfg = cfg["train"]
    tr_loader, va_loader, te_loader = make_loaders(
        tr_items, va_items, te_items,
        label2id, cfg, cfg["expert"]["type"],
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 4)),
    )

    opt = make_optimizer(model, float(train_cfg["lr"]), float(train_cfg.get("weight_decay", 0.0)))
    total_steps = int(train_cfg["epochs"]) * max(1, len(tr_loader))
    sch = make_scheduler(opt, train_cfg.get("scheduler", "cosine"), total_steps, float(train_cfg.get("warmup_ratio", 0.0)))

    best_metric = -1e9
    best_path = os.path.join(run_dir, "checkpoints", "best.pt")
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    es = train_cfg.get("early_stop", {"enabled": False})
    patience = int(es.get("patience", 10))
    metric_name = es.get("metric", "uar")
    bad = 0

    grad_clip = float(train_cfg.get("grad_clip", 0.0))

    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        pbar = tqdm(tr_loader, desc=f"[{exp_name} fold{fold}] epoch {epoch}")

        for batch in pbar:
            y = batch["y"].to(device)

            for k in list(batch.keys()):
                if k.startswith("x_"):
                    batch[k] = batch[k].to(device)

            logits = model(batch)
            loss = loss_fn(logits, y) if callable(loss_fn) else loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()
            sch.step()

            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

        val_out = eval_and_dump(model, va_loader, device, num_classes, save_path=None)
        cur = float(val_out.get(metric_name, val_out["uar"]))

        with open(os.path.join(run_dir, "val_metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, **val_out}, ensure_ascii=False) + "\n")

        if cur > best_metric:
            best_metric = cur
            torch.save({"model": model.state_dict(), "cfg": cfg, "label2id": label2id, "fold_id": int(fold)}, best_path)
            bad = 0
        else:
            bad += 1

        if es.get("enabled", False) and bad >= patience:
            break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    save_preds = cfg.get("log", {}).get("save_preds", True)
    val_pred_path = os.path.join(run_dir, "preds", "val_logits.npz") if save_preds else None
    te_pred_path = os.path.join(run_dir, "preds", "test_logits.npz") if save_preds else None

    val_out = eval_and_dump(model, va_loader, device, num_classes, save_path=val_pred_path)
    te_out = eval_and_dump(model, te_loader, device, num_classes, save_path=te_pred_path)

    with open(os.path.join(run_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"val": val_out, "test": te_out}, f, ensure_ascii=False, indent=2)

    return {
        "run_dir": run_dir,
        "best_ckpt": best_path,
        "metrics": {"val": val_out, "test": te_out},
    }


def train_from_yaml(
    expert_yaml_path: str,
    dataset_yaml_path: str,
    fold: int = 0,
    device: Optional[str] = None,
    label_scheme: Optional[str] = None,
) -> Dict[str, Any]:
    expert_cfg = load_yaml(expert_yaml_path)
    dataset_cfg = load_yaml(dataset_yaml_path)
    if label_scheme is not None and str(label_scheme).strip():
        dataset_cfg["label_scheme"] = str(label_scheme).strip()
    return train_from_dict(expert_cfg, dataset_cfg, fold=fold, device=device)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # pick an existing example yaml as default
    ap.add_argument("--config", default="audio_whisper_cnn_attention.yaml", help="expert config yaml")
    ap.add_argument("--dataset", default="configs/datasets/iemocap.yaml", help="dataset config yaml")
    ap.add_argument(
        "--label_scheme",
        default=None,
        help="Override dataset label_scheme (e.g., 4way or 6way). If omitted, use YAML.",
    )
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    out = train_from_yaml(args.config, args.dataset, fold=args.fold, device=args.device, label_scheme=args.label_scheme)
    print("Done:", out["metrics"]["test"])


if __name__ == "__main__":
    main()
