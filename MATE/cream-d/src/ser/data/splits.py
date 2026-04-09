from __future__ import annotations

"""
【文件作用】数据划分逻辑：GroupKFold 与 repeated-holdout（speaker-level，无泄漏）。

（如果你在 IDE 里阅读代码：先看本段，再往下看实现。）
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.model_selection import GroupKFold


def make_group_kfold_splits(
    items: List[Dict[str, Any]],
    group_key: str,
    n_splits: int,
    seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # deterministic order
    rng = np.random.RandomState(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    shuffled = [items[i] for i in idx]

    groups = [it[group_key] for it in shuffled]
    gkf = GroupKFold(n_splits=n_splits)

    folds = []
    for fold_id, (tr_idx, te_idx) in enumerate(gkf.split(np.zeros(len(shuffled)), groups=groups)):
        fold = {"fold": fold_id, "train_idx": tr_idx.tolist(), "test_idx": te_idx.tolist()}
        folds.append(fold)

    return folds, shuffled


def make_repeated_holdout_splits(
    items: List[Dict[str, Any]],
    group_key: str,
    repeats: int,
    seed: int,
    ratio_train: float,
    ratio_val: float,
    ratio_test: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Repeated random speaker-level holdout.
    Each repeat creates speaker-disjoint train/val/test splits by ratios.

    Returns:
      folds: list of {"fold", "train_idx", "val_idx", "test_idx"} indices w.r.t. shuffled list
      shuffled: deterministic shuffled items list
    """
    s = float(ratio_train) + float(ratio_val) + float(ratio_test)
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {s}")

    # deterministic order of samples (so idx is stable)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    shuffled = [items[i] for i in idx]
    groups = [str(it[group_key]) for it in shuffled]

    speakers = sorted(set(groups))
    n_spk = len(speakers)
    if n_spk < 3:
        raise ValueError(f"Need >=3 speakers for train/val/test, got {n_spk}")

    folds = []
    for fold_id in range(int(repeats)):
        rng2 = np.random.RandomState(seed + 1000 + fold_id)
        spk = speakers.copy()
        rng2.shuffle(spk)

        n_tr = int(round(ratio_train * n_spk))
        n_va = int(round(ratio_val * n_spk))

        # ensure at least 1 speaker in each split
        n_tr = max(1, min(n_tr, n_spk - 2))
        n_va = max(1, min(n_va, n_spk - n_tr - 1))
        n_te = n_spk - n_tr - n_va
        if n_te < 1:
            n_te = 1
            # adjust val first, then train if needed
            if n_va > 1:
                n_va -= 1
            else:
                n_tr = max(1, n_tr - 1)

        tr_spk = set(spk[:n_tr])
        va_spk = set(spk[n_tr:n_tr + n_va])
        te_spk = set(spk[n_tr + n_va:])

        # leak check
        if tr_spk & va_spk or tr_spk & te_spk or va_spk & te_spk:
            raise RuntimeError("Speaker leakage detected (should never happen).")

        train_idx = [i for i, g in enumerate(groups) if g in tr_spk]
        val_idx   = [i for i, g in enumerate(groups) if g in va_spk]
        test_idx  = [i for i, g in enumerate(groups) if g in te_spk]

        folds.append({
            "fold": fold_id,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "num_speakers": {"train": len(tr_spk), "val": len(va_spk), "test": len(te_spk)},
        })

    return folds, shuffled
