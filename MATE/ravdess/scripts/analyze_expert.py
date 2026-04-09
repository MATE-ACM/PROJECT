# scripts/analyze_expert.py
from __future__ import annotations

"""
【文件作用】专家诊断导出（逐样本）
- 读取 run_dir/checkpoints/best.pt
- 对指定 split(train/val/test) 导出：
  loss / confidence / margin_top2 / entropy （可选 MC-dropout 指标）
- 可选导出 pooled embedding（如果模型支持 forward_with_extras）
- 输出到：run_dir/analysis/{split}_samples.csv 以及 {split}_pooled.npz

【重要设计】不“自作主张”重切分：
1) 优先读取 run_dir/splits.json（推荐：训练时写入）
2) 如果没有 splits.json，才根据 cfg["protocol"] 复现切分
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.ser.data.manifest import read_jsonl
from src.ser.experts import build_expert



RUN_DIR = r""  # 例如 r"fold_0"
SPLITS_TO_EXPORT = ["val", "test"]  # 一般只关心 val/test
MC_SAMPLES = 0                     # 0=关闭；>0 开启 MC-dropout
SAVE_EMBEDDINGS = True             # True 导出 pooled embedding（若模型支持）
BATCH_SIZE = 64
NUM_WORKERS = 0
DEVICE = None  # None=自动 cuda/cpu


def _get_manifest_path(cfg: Dict[str, Any]) -> str:
    if "manifest_path" in cfg:
        return cfg["manifest_path"]
    if "manifest" in cfg:
        return cfg["manifest"]
    raise KeyError(" manifest 字段，请统一一下配置键名。")


def _get_label_maps(cfg: Dict[str, Any], ckpt: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[int, str]]:
    # 兼容：旧 ckpt 里可能有 label2id
    if "label2id" in ckpt:
        label2id = ckpt["label2id"]
        id2label = {v: k for k, v in label2id.items()}
        return label2id, id2label

    # 推荐：从 cfg["labels"] 推导
    labels = cfg.get("labels", None)
    if not labels:
        raise KeyError("ckpt 没有 label2id，同时 cfg 里也没有 labels。至少保留一个。")
    label2id = {lb: i for i, lb in enumerate(labels)}
    id2label = {i: lb for lb, i in label2id.items()}
    return label2id, id2label


def _parse_fold_id(run_dir: str) -> int:
    base = os.path.basename(os.path.normpath(run_dir))
    if base.startswith("fold_"):
        try:
            return int(base.split("_", 1)[1])
        except Exception:
            return 0
    return 0


def _split_train_val_by_group(items: List[Dict[str, Any]], group_key: str, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    groups = sorted({it.get(group_key) for it in items if it.get(group_key) is not None})
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * val_ratio)))
    val_g = set(groups[:n_val])
    tr, va = [], []
    for it in items:
        (va if it.get(group_key) in val_g else tr).append(it)
    return tr, va


def _enable_dropout(m: torch.nn.Module):
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.train()


@torch.no_grad()
def _mc_metrics(model: torch.nn.Module, batch: Dict[str, Any], mc_samples: int) -> Dict[str, torch.Tensor]:
    model.eval()
    _enable_dropout(model)

    probs = []
    for _ in range(mc_samples):
        logits = model(batch)
        probs.append(torch.softmax(logits, dim=-1).unsqueeze(0))  # [1,B,C]
    probs = torch.cat(probs, dim=0)  # [S,B,C]

    mean_p = probs.mean(dim=0)
    pred_entropy = -(mean_p * (mean_p.clamp_min(1e-9).log())).sum(dim=-1)
    ent_each = -(probs * (probs.clamp_min(1e-9).log())).sum(dim=-1)
    expected_entropy = ent_each.mean(dim=0)
    mutual_info = pred_entropy - expected_entropy
    return {"mc_pred_entropy": pred_entropy, "mc_expected_entropy": expected_entropy, "mc_mutual_info": mutual_info}


def _load_splits_if_exist(run_dir: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(run_dir, "splits.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_items_by_utt_id(items: List[Dict[str, Any]], utt_ids: List[str]) -> List[Dict[str, Any]]:
    lut = {str(it["utt_id"]): it for it in items}
    out = []
    miss = 0
    for u in utt_ids:
        it = lut.get(str(u), None)
        if it is None:
            miss += 1
            continue
        out.append(it)
    if miss > 0:
        print(f"[WARN] splits.json 有 {miss} 条 utt_id 在 manifest 里找不到（可能 manifest 变了）。")
    return out


def _rebuild_split_from_cfg(items: List[Dict[str, Any]], cfg: Dict[str, Any], fold_id: int):
    """
    兜底方案：当 run_dir 没有 splits.json 时，用 cfg["protocol"] 复现切分
    - 支持 repeated_holdout / group_kfold
    """
    from src.ser.data.splits import make_repeated_holdout_splits, make_group_kfold_splits

    protocol = cfg.get("protocol", {})
    ptype = protocol.get("type", cfg.get("protocol_type", "group_kfold"))
    seed = int(protocol.get("seed", cfg.get("seed", 0)))
    group_key = protocol.get("group_key", "speaker_id")

    if ptype == "repeated_holdout":
        repeats = int(protocol.get("repeats", cfg.get("folds", 5)))
        ratio_train = float(protocol.get("ratio_train", 0.8))
        ratio_val = float(protocol.get("ratio_val", 0.1))
        ratio_test = float(protocol.get("ratio_test", 0.1))

        out = make_repeated_holdout_splits(
            items,
            group_key=group_key,
            repeats=repeats,
            seed=seed,
            ratio_train=ratio_train,
            ratio_val=ratio_val,
            ratio_test=ratio_test,
        )
        # 兼容两种返回：(fold_defs, items_shuffled) 或 仅 fold_defs
        if isinstance(out, tuple) and len(out) == 2:
            fold_defs, items2 = out
        else:
            fold_defs, items2 = out, items

        fd = fold_defs[fold_id]
        tr = [items2[i] for i in fd["train_idx"]]
        va = [items2[i] for i in fd["val_idx"]]
        te = [items2[i] for i in fd["test_idx"]]
        return tr, va, te

    # 默认 group_kfold
    n_splits = int(protocol.get("n_splits", cfg.get("folds", 5)))
    val_ratio_in_train = float(protocol.get("val_ratio_in_train", cfg.get("val_ratio", 0.1)))

    out = make_group_kfold_splits(items, group_key=group_key, n_splits=n_splits, seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        fold_defs, items2 = out
    else:
        fold_defs, items2 = out, items

    fd = fold_defs[fold_id]
    if isinstance(fd, dict) and "train_idx" in fd:
        tr_all = [items2[i] for i in fd["train_idx"]]
        te = [items2[i] for i in fd["test_idx"]]
    else:
        # 兼容老返回：[(tr_items, te_items), ...]
        tr_all, te = fold_defs[fold_id]

    tr, va = _split_train_val_by_group(tr_all, group_key=group_key, val_ratio=val_ratio_in_train, seed=seed + fold_id)
    return tr, va, te


def analyze_run_dir(
    run_dir: str,
    split: str = "test",
    device: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    mc_samples: int = 0,
    save_embeddings: bool = True,
):
    assert split in ["train", "val", "test"]

    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]

    label2id, id2label = _get_label_maps(cfg, ckpt)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    items = read_jsonl(_get_manifest_path(cfg))
    fold_id = _parse_fold_id(run_dir)

    # 1) 优先读 splits.json（强烈推荐）
    sp = _load_splits_if_exist(run_dir)
    if sp is not None:
        tr_items = _select_items_by_utt_id(items, sp["train_utt_id"])
        va_items = _select_items_by_utt_id(items, sp["val_utt_id"])
        te_items = _select_items_by_utt_id(items, sp["test_utt_id"])
    else:
        # 2) 兜底：按 cfg["protocol"] 复现
        tr_items, va_items, te_items = _rebuild_split_from_cfg(items, cfg, fold_id=fold_id)

    items_split = {"train": tr_items, "val": va_items, "test": te_items}[split]

    from src.ser.data.dataloaders import make_loaders
    dl_tr, dl_va, dl_te = make_loaders(
        items_split, items_split, items_split,
        label2id, cfg, cfg["expert"]["type"],
        batch_size=batch_size, num_workers=num_workers,
    )
    loader = dl_te  # make_loaders 的第三个就返回 dl_te，这里借用它来跑当前 split

    model = build_expert(cfg["expert"]).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    rows = []
    pooled_all = []
    utt_all = []

    for batch in tqdm(loader, desc=f"analyze[{split}]"):
        for k in list(batch.keys()):
            if k.startswith("x_") or k == "y":
                batch[k] = batch[k].to(device)

        y = batch["y"]

        pooled = None
        if hasattr(model, "forward_with_extras"):
            logits, pooled, _attn = model.forward_with_extras(batch)  # type: ignore
        else:
            logits = model(batch)

        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)

        top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]) if top2.shape[-1] == 2 else torch.zeros_like(conf)

        entropy = -(probs * (probs.clamp_min(1e-9).log())).sum(dim=-1)
        loss = loss_fn(logits, y)
        correct = (pred == y)

        mc = {}
        if mc_samples and mc_samples > 0:
            mc = _mc_metrics(model, batch, mc_samples=mc_samples)

        utt_id = batch["utt_id"]
        for i in range(len(utt_id)):
            yt = int(y[i].detach().cpu().item())
            yp = int(pred[i].detach().cpu().item())
            row = {
                "utt_id": utt_id[i],
                "y_true": yt,
                "y_true_label": id2label.get(yt, str(yt)),
                "y_pred": yp,
                "y_pred_label": id2label.get(yp, str(yp)),
                "correct": bool(correct[i].detach().cpu().item()),
                "loss": float(loss[i].detach().cpu().item()),
                "confidence": float(conf[i].detach().cpu().item()),
                "margin_top2": float(margin[i].detach().cpu().item()),
                "entropy": float(entropy[i].detach().cpu().item()),
            }
            for kk, vv in mc.items():
                row[kk] = float(vv[i].detach().cpu().item())
            rows.append(row)

        if pooled is not None and save_embeddings:
            pooled_all.append(pooled.detach().cpu().numpy())
            utt_all.extend(list(utt_id))

    out_dir = os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{split}_samples.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")

    if save_embeddings and pooled_all:
        emb = np.concatenate(pooled_all, axis=0)
        np.savez_compressed(os.path.join(out_dir, f"{split}_pooled.npz"), utt_id=np.array(utt_all), pooled=emb)

    print("Saved:", csv_path)


def analyze_after_train(
    run_dir: str,
    splits: Optional[List[str]] = None,
    mc_samples: int = 0,
    save_embeddings: bool = True,
    device: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 0,
):
    if splits is None:
        splits = ["val", "test"]
    for sp in splits:
        analyze_run_dir(
            run_dir=run_dir,
            split=sp,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            mc_samples=mc_samples,
            save_embeddings=save_embeddings,
        )


def main():
    if not RUN_DIR:
        raise ValueError("请先在脚本顶部填 RUN_DIR（指向某个 fold_x 的 run_dir）。")

    analyze_after_train(
        run_dir=RUN_DIR,
        splits=SPLITS_TO_EXPORT,
        mc_samples=MC_SAMPLES,
        save_embeddings=SAVE_EMBEDDINGS,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )


if __name__ == "__main__":
    main()
