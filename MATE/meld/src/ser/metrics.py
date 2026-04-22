from __future__ import annotations

from typing import Dict, Any
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

def war(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())

def uar(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.diag(cm) / cm.sum(axis=1).clip(min=1)
    return float(np.nanmean(recall))

def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    return f1_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

def summarize(y_true: np.ndarray, logits: np.ndarray, num_classes: int) -> Dict[str, Any]:
    y_pred = logits.argmax(axis=1)
    out = {
        "war": war(y_true, y_pred),
        "uar": uar(y_true, y_pred, num_classes),
        "macro_f1": macro_f1(y_true, y_pred),
        "weighted_f1": weighted_f1(y_true, y_pred),
        "per_class_f1": per_class_f1(y_true, y_pred, num_classes).tolist(),
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
    }
    return out