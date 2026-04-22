from __future__ import annotations

"""Calibration algorithms and evaluation helpers."""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import json
import numpy as np
import torch
import torch.nn.functional as F

# NOTE:
#

@dataclass
class Calibrator:
    """校准器基类，定义统一接口"""
    name: str

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "Calibrator":
        """在验证集上拟合校准参数"""
        raise NotImplementedError

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        """应用校准参数转换 logits (用于 Logit Calibrators)"""
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        """序列化参数"""
        return {"name": self.name}

    @staticmethod
    def from_state_dict(sd: Dict[str, Any]) -> "Calibrator":
        """从字典加载校准器"""
        name = sd.get("name", "")
        if name == "temperature":
            c = TemperatureScaling()
            c.temperature = float(sd["temperature"])
            return c
        if name == "vector":
            c = VectorScaling(num_classes=int(sd["num_classes"]))
            c.scale = np.array(sd["scale"], dtype=np.float32)
            c.bias = np.array(sd["bias"], dtype=np.float32)
            return c
        if name == "conf_isotonic":
            c = ConfidenceIsotonic()
            c.x_thresholds = np.array(sd["x_thresholds"], dtype=np.float32)
            c.y_thresholds = np.array(sd["y_thresholds"], dtype=np.float32)
            return c
        if name == "conf_linear":
            c = ConfidenceLinear()
            c.a = float(sd["a"])
            c.b = float(sd["b"])
            return c
        raise KeyError(f"Unknown calibrator name in state_dict: {name}")

# ------------------------
# Logit calibrators (Ours / Standard)
# ------------------------

class TemperatureScaling(Calibrator):
    """
    温度缩放 (Platt Scaling 的一种变体)
    公式: logits_calibrated = logits / T
    只有一个参数 T (temperature)。
    """

    def __init__(self, init_T: float = 1.0, max_iter: int = 2000, lr: float = 0.05, device: str | None = None):
        super().__init__(name="temperature")
        self.temperature = float(init_T)
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaling":
        """使用 LBFGS 优化器寻找最小化 NLL 的最佳温度 T"""
        logits_t = torch.tensor(logits, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_true, dtype=torch.long, device=self.device)

        log_T = torch.tensor([np.log(max(self.temperature, 1e-3))], dtype=torch.float32, device=self.device,
                             requires_grad=True)

        opt = torch.optim.LBFGS([log_T], lr=self.lr, max_iter=50, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            T = torch.exp(log_T).clamp(min=1e-3, max=1e3)
            loss = F.cross_entropy(logits_t / T, y_t)
            loss.backward()
            return loss

        for _ in range(max(1, self.max_iter // 50)):
            opt.step(closure)

        self.temperature = float(torch.exp(log_T).detach().cpu().item())
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return logits / max(self.temperature, 1e-6)

    def state_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "temperature": float(self.temperature)}

class VectorScaling(Calibrator):
    """
    向量缩放 (Matrix/Vector Scaling)
    公式: logits' = logits * scale + bias (此处为 element-wise，即对角矩阵)
    为每个类别学习独立的缩放系数和偏置。
    """

    def __init__(self, num_classes: int, max_iter: int = 3000, lr: float = 1e-2, weight_decay: float = 1e-4,
                 device: str | None = None):
        super().__init__(name="vector")
        self.num_classes = int(num_classes)
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = np.ones((self.num_classes,), dtype=np.float32)
        self.bias = np.zeros((self.num_classes,), dtype=np.float32)

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "VectorScaling":
        """使用 AdamW 优化 per-class 的 scale 和 bias"""
        logits_t = torch.tensor(logits, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_true, dtype=torch.long, device=self.device)

        scale = torch.ones((self.num_classes,), dtype=torch.float32, device=self.device, requires_grad=True)
        bias = torch.zeros((self.num_classes,), dtype=torch.float32, device=self.device, requires_grad=True)

        opt = torch.optim.AdamW([scale, bias], lr=self.lr, weight_decay=self.weight_decay)

        best = float("inf")
        best_state = None
        for _ in range(self.max_iter):
            opt.zero_grad(set_to_none=True)
            cal = logits_t * scale + bias
            loss = F.cross_entropy(cal, y_t)
            loss.backward()
            opt.step()
            if loss.item() < best:
                best = loss.item()
                best_state = (scale.detach().clone(), bias.detach().clone())

        if best_state is not None:
            scale, bias = best_state
        self.scale = scale.detach().cpu().numpy().astype(np.float32)
        self.bias = bias.detach().cpu().numpy().astype(np.float32)
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return logits * self.scale[None, :] + self.bias[None, :]

    def state_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "num_classes": self.num_classes, "scale": self.scale.tolist(),
                "bias": self.bias.tolist()}

# -----------------------------------
# Confidence calibrators (MoCaE-like)
# -----------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numpy版 Softmax，带数值稳定性处理"""
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.clip(ez.sum(axis=1, keepdims=True), 1e-12, None)

class ConfidenceIsotonic(Calibrator):
    """
    保序回归 (Isotonic Regression)
    不假设参数形式，只假设 Correctness 是 Confidence 的单调非减函数。
    输入: 原始置信度 (0-1)
    输出: 校准后的 P(Correct)
    """

    def __init__(self):
        super().__init__(name="conf_isotonic")
        self.x_thresholds = None
        self.y_thresholds = None

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "ConfidenceIsotonic":
        from sklearn.isotonic import IsotonicRegression
        p = _softmax(logits)
        conf = p.max(axis=1).astype(np.float64)
        pred = p.argmax(axis=1)
        correct = (pred == y_true).astype(np.float64)

        ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        ir.fit(conf, correct)

        self.x_thresholds = np.array(ir.X_thresholds_, dtype=np.float32)
        self.y_thresholds = np.array(ir.y_thresholds_, dtype=np.float32)
        return self

    def transform_confidence(self, conf: np.ndarray) -> np.ndarray:
        """输入原始置信度标量，输出校准后的真实准确率估计"""
        conf = np.asarray(conf, dtype=np.float32)
        return np.interp(conf, self.x_thresholds, self.y_thresholds).astype(np.float32)

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return logits

    def state_dict(self) -> Dict[str, Any]:
        return {"name": self.name,
                "x_thresholds": [] if self.x_thresholds is None else self.x_thresholds.tolist(),
                "y_thresholds": [] if self.y_thresholds is None else self.y_thresholds.tolist()}

class ConfidenceLinear(Calibrator):
    """
    线性回归校准
    假设 P(Correct) = a * Confidence + b
    比保序回归更强硬的假设，但在数据稀疏时更稳定。
    """

    def __init__(self):
        super().__init__(name="conf_linear")
        self.a = 1.0
        self.b = 0.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "ConfidenceLinear":
        p = _softmax(logits)
        conf = p.max(axis=1).astype(np.float64)
        pred = p.argmax(axis=1)
        correct = (pred == y_true).astype(np.float64)

        X = np.stack([conf, np.ones_like(conf)], axis=1)
        theta, _, _, _ = np.linalg.lstsq(X, correct, rcond=None)
        a, b = float(theta[0]), float(theta[1])

        if a < 0:
            a = 0.0
            b = float(correct.mean())
        self.a, self.b = a, b
        return self

    def transform_confidence(self, conf: np.ndarray) -> np.ndarray:
        conf = np.asarray(conf, dtype=np.float32)
        out = self.a * conf + self.b
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return logits

    def state_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "a": float(self.a), "b": float(self.b)}

def save_calibrator(path: str, calibrator: Calibrator) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calibrator.state_dict(), f, ensure_ascii=False, indent=2)

def load_calibrator(path: str) -> Calibrator:
    with open(path, "r", encoding="utf-8") as f:
        sd = json.load(f)
    return Calibrator.from_state_dict(sd)

## Calibration metrics (ECE/NLL/Brier) + reliability diagram utils

def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.clip(ez.sum(axis=1, keepdims=True), 1e-12, None)

def nll(logits: np.ndarray, y_true: np.ndarray) -> float:
    """负对数似然 (Negative Log Likelihood)。衡量概率分布的质量，对错误且自信的预测惩罚大。"""
    p = softmax(logits)
    idx = (np.arange(len(y_true)), y_true.astype(np.int64))
    return float(-np.log(np.clip(p[idx], 1e-12, 1.0)).mean())

def brier_multiclass(logits: np.ndarray, y_true: np.ndarray, num_classes: int) -> float:
    """Brier Score (Multiclass)。类似于 MSE，衡量预测概率向量与 One-hot 标签向量的欧氏距离。"""
    p = softmax(logits)
    y = np.zeros_like(p)
    y[np.arange(len(y_true)), y_true.astype(np.int64)] = 1.0
    return float(np.mean(np.sum((p - y) ** 2, axis=1)))

def ece_from_confidence(conf: np.ndarray, correct01: np.ndarray, n_bins: int = 15) -> float:
    """
    计算 ECE (Expected Calibration Error)
    公式: ECE = sum( bin_weight * | avg_confidence_in_bin - avg_accuracy_in_bin | )
    """
    conf = np.asarray(conf, dtype=np.float32)
    correct01 = np.asarray(correct01, dtype=np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        acc = float(correct01[m].mean())
        avg_conf = float(conf[m].mean())
        ece += (m.sum() / N) * abs(acc - avg_conf)
    return float(ece)

def ece_multiclass(logits: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """针对多分类 Logits 计算 ECE (基于 Max Probability)"""
    p = softmax(logits)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)
    return ece_from_confidence(conf, correct, n_bins=n_bins)

def reliability_curve_from_confidence(conf: np.ndarray, correct01: np.ndarray, n_bins: int = 15) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """生成绘制 Reliability Diagram 所需的数据点 (X:置信度, Y:准确率)"""
    conf = np.asarray(conf, dtype=np.float32)
    correct01 = np.asarray(correct01, dtype=np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        centers.append((lo + hi) / 2)
        accs.append(float(correct01[m].mean()))
        confs.append(float(conf[m].mean()))
    return np.array(centers), np.array(accs), np.array(confs)

def reliability_curve_from_logits(logits: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """从 Logits 生成 Reliability Diagram 数据"""
    p = softmax(logits)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)
    return reliability_curve_from_confidence(conf, correct, n_bins=n_bins)

def summarize_calibration(logits: np.ndarray, y_true: np.ndarray, num_classes: int, n_bins: int = 15) -> Dict[str, Any]:
    """计算所有校准指标的汇总字典"""
    return {
        "nll": nll(logits, y_true),
        "brier": brier_multiclass(logits, y_true, num_classes),
        "ece": ece_multiclass(logits, y_true, n_bins=n_bins),
        "n_bins": int(n_bins),
    }

# ----------------------------
# Confidence/quality calibration metrics (MoCaE-like)
#   target is correctness (0/1), not class probabilities.
# ----------------------------

def brier_binary(conf: np.ndarray, correct01: np.ndarray) -> float:
    """二元 Brier Score: 衡量 (置信度 - 正确性) 的差距"""
    conf = np.asarray(conf, dtype=np.float32)
    correct01 = np.asarray(correct01, dtype=np.float32)
    return float(np.mean((conf - correct01) ** 2))

def nll_binary(conf: np.ndarray, correct01: np.ndarray) -> float:
    """二元 NLL: 衡量校准后的置信度是否有效地预测了“预测正确”这件事"""
    conf = np.asarray(conf, dtype=np.float32)
    correct01 = np.asarray(correct01, dtype=np.float32)
    conf = np.clip(conf, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(correct01 * np.log(conf) + (1.0 - correct01) * np.log(1.0 - conf)))

def summarize_quality(conf: np.ndarray, correct01: np.ndarray, n_bins: int = 15) -> Dict[str, Any]:
    """汇总置信度质量指标 (专用于 MoCaE 路由器的评估)"""
    return {
        "q_nll": nll_binary(conf, correct01),
        "q_brier": brier_binary(conf, correct01),
        "q_ece": ece_from_confidence(conf, correct01, n_bins=n_bins),
        "n_bins": int(n_bins),
    }