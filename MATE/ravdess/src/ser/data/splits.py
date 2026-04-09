from __future__ import annotations

"""
【文件作用】数据划分逻辑：GroupKFold 与 repeated-holdout（speaker-level，无泄漏）。

（如果你在 IDE 里阅读代码：先看本段，再往下看实现。）
"""

from typing import Dict, List, Any, Tuple

import re
import numpy as np
from sklearn.model_selection import GroupKFold


_RE_INT = re.compile(r"(\d+)")


def _group_sort_key(g: Any) -> tuple:
    """Robust stable sort key for group ids.

    - If g is int-like -> sort numerically.
    - If g is str and contains digits -> sort by the last digit group numerically.
      (e.g., "Actor_01" -> 1)
    - Otherwise -> sort lexicographically.

    Returns a tuple so mixed types are still comparable.
    """
    if g is None:
        return (2, "")

    # numeric first
    try:
        if isinstance(g, (int, np.integer)):
            return (0, int(g))
        if isinstance(g, (float, np.floating)) and float(g).is_integer():
            return (0, int(g))
    except Exception:
        pass

    s = str(g)
    m = list(_RE_INT.finditer(s))
    if m:
        # use last number as a proxy for id
        try:
            return (0, int(m[-1].group(1)))
        except Exception:
            pass
    return (1, s)


def _shuffle_items(items: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(int(seed))
    idx = np.arange(len(items))
    rng.shuffle(idx)
    return [items[i] for i in idx]


def _validate_groups(groups: List[Any], n_splits: int, group_key: str) -> List[Any]:
    uniq = sorted({g for g in groups if g is not None}, key=_group_sort_key)
    if len(uniq) == 0:
        raise ValueError(f"No valid groups found by group_key='{group_key}'.")
    if len(uniq) < int(n_splits):
        raise ValueError(
            f"Need >= n_splits unique groups. got groups={len(uniq)}, n_splits={n_splits} (group_key='{group_key}')."
        )
    return uniq


def make_group_kfold_splits(
    items: List[Dict[str, Any]],
    group_key: str,
    n_splits: int,
    seed: int,
    method: str = "sklearn",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Speaker/group-disjoint K-Fold splits.

    This function is used by training/analyze scripts.

    Args:
        items: manifest records.
        group_key: field name in each item, e.g. speaker_id / actor_id.
        n_splits: number of folds.
        seed: seed controlling deterministic sample order.
        method:
            - "sklearn" (default): sklearn.model_selection.GroupKFold (balanced by sample count)
            - "blocks": sort unique groups, then split into n_splits contiguous blocks
                        (RAVDESS common: 24 actors / 6 folds => 4 actors per fold)
            - "bucket": shuffle groups by seed then distribute round-robin into n_splits buckets

    Returns:
        folds: list of {fold, train_idx, test_idx, (optional) test_groups}
        shuffled: deterministic shuffled items list, indices refer to this list.
    """

    shuffled = _shuffle_items(items, seed=int(seed))

    # be strict: group_key must exist for ALL items (avoid silent leakage)
    groups = []
    missing = 0
    none_cnt = 0
    for it in shuffled:
        if group_key not in it:
            missing += 1
            continue
        g = it[group_key]
        if g is None:
            none_cnt += 1
            continue
        groups.append(g)

    if missing > 0:
        raise KeyError(f"{missing} items are missing group_key='{group_key}'. Fix your manifest.")
    if none_cnt > 0:
        raise ValueError(f"{none_cnt} items have group_key='{group_key}' == None. Fix your manifest.")

    method = str(method).lower().strip()
    if method in ["sklearn", "groupkfold", "gkf"]:
        _validate_groups(groups, int(n_splits), group_key)
        gkf = GroupKFold(n_splits=int(n_splits))
        folds = []
        for fold_id, (tr_idx, te_idx) in enumerate(gkf.split(np.zeros(len(shuffled)), groups=groups)):
            folds.append({"fold": int(fold_id), "train_idx": tr_idx.tolist(), "test_idx": te_idx.tolist()})
        return folds, shuffled

    # group-level splits (equal group counts as much as possible)
    uniq = _validate_groups(groups, int(n_splits), group_key)

    if method in ["blocks", "actor_blocks", "ravdess_actor_blocks", "contiguous"]:
        # e.g. [1..24] -> 6 chunks: [1..4], [5..8], ...
        chunks = np.array_split(np.array(uniq, dtype=object), int(n_splits))
        folds = []
        for fold_id, ch in enumerate(chunks):
            te_groups = set(ch.tolist())
            te_mask = np.isin(np.array(groups, dtype=object), list(te_groups))
            te_idx = np.where(te_mask)[0]
            tr_idx = np.where(~te_mask)[0]
            folds.append(
                {
                    "fold": int(fold_id),
                    "train_idx": tr_idx.tolist(),
                    "test_idx": te_idx.tolist(),
                    "test_groups": sorted(list(te_groups), key=_group_sort_key),
                }
            )
        return folds, shuffled

    if method in ["bucket", "shuffle_bucket", "roundrobin"]:
        rng = np.random.RandomState(int(seed) + 2027)
        spk = uniq.copy()
        rng.shuffle(spk)
        buckets = [set() for _ in range(int(n_splits))]
        for i, g in enumerate(spk):
            buckets[i % int(n_splits)].add(g)

        folds = []
        for fold_id, te_groups in enumerate(buckets):
            te_mask = np.isin(np.array(groups, dtype=object), list(te_groups))
            te_idx = np.where(te_mask)[0]
            tr_idx = np.where(~te_mask)[0]
            folds.append(
                {
                    "fold": int(fold_id),
                    "train_idx": tr_idx.tolist(),
                    "test_idx": te_idx.tolist(),
                    "test_groups": sorted(list(te_groups), key=_group_sort_key),
                }
            )
        return folds, shuffled

    raise KeyError(f"Unknown method for make_group_kfold_splits: {method}")


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
