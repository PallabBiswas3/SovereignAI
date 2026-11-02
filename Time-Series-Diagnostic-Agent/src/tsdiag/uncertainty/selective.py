"""Selective-prediction and calibration metrics (numpy only).

Reporting standard for every domain: accuracy/risk *at a coverage*, not a
single accuracy number.  A tool that abstains on 69% of inputs is only
meaningful if the answered 31% are demonstrably reliable; the risk-coverage
curve is how you show that.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .conformal import _labels, _probabilities

__all__ = [
    "softmax",
    "fit_temperature",
    "expected_calibration_error",
    "risk_coverage_curve",
    "aurc",
    "accuracy_at_coverage",
    "selective_summary",
]


def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    logits = _logits(logits)
    z = np.asarray(logits, dtype=float) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _logits(logits):
    z = np.asarray(logits, dtype=float)
    if z.ndim != 2 or not z.shape[1] or not np.isfinite(z).all():
        raise ValueError("logits must be a finite (n, K) array with at least one class")
    return z


def _nll(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-log_p[np.arange(y.size), y].mean())


def fit_temperature(logits, y, low: float = 0.05, high: float = 10.0) -> float:
    """Temperature scaling: the single T minimising NLL on a calibration set.

    Fit on calibration units only (never on the frozen test set).
    """
    logits = _logits(logits)
    y = _labels(y, len(logits), logits.shape[1])
    if not len(y) or not np.isfinite([low, high]).all() or not 0 < low < high:
        raise ValueError("need nonempty calibration labels and finite 0 < low < high")
    grid = np.exp(np.linspace(np.log(low), np.log(high), 200))
    if low <= 1 <= high:
        grid = np.append(grid, 1.0)
    best = grid[int(np.argmin([_nll(logits, y, t) for t in grid]))]
    fine = np.append(np.linspace(max(low, best * 0.9), min(high, best * 1.1), 60), best)
    return float(fine[int(np.argmin([_nll(logits, y, t) for t in fine]))])


def expected_calibration_error(probs, y, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins."""
    probs = _probabilities(probs)
    y = _labels(y, len(probs), probs.shape[1])
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if not y.size:
        return float("nan")
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def risk_coverage_curve(confidence, correct) -> Tuple[np.ndarray, np.ndarray]:
    """Answer the most confident samples first.

    Returns (coverage, risk) where risk[i] is the error rate among the
    coverage[i] most confident samples.
    """
    c = np.asarray(confidence, dtype=float).ravel()
    ok = np.asarray(correct).ravel()
    if (c.size != ok.size or not np.isfinite(c).all()
            or not np.isin(ok, [0, 1]).all()):
        raise ValueError("confidence and correctness must be finite and aligned; correctness is boolean")
    ok = ok.astype(bool)
    if not c.size:
        return np.array([], dtype=float), np.array([], dtype=float)
    order = np.argsort(-c, kind="stable")
    k = np.arange(1, ok.size + 1)
    coverage = k / ok.size
    risk = 1.0 - np.cumsum(ok[order]) / k
    return coverage, risk


def aurc(confidence, correct) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    _, risk = risk_coverage_curve(confidence, correct)
    return float(risk.mean()) if risk.size else float("nan")


def accuracy_at_coverage(confidence, correct, coverage: float) -> float:
    """Accuracy when only the top `coverage` fraction (by confidence) is answered."""
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    cov, risk = risk_coverage_curve(confidence, correct)
    if not cov.size:
        return float("nan")
    i = min(int(np.ceil(coverage * cov.size)) - 1, cov.size - 1)
    return float(1.0 - risk[i])


def selective_summary(label, y, abstain) -> Dict[str, float]:
    """Headline numbers for a model that may abstain (label = -1 or abstain mask)."""
    label, y, abstain = (np.asarray(a) for a in (label, y, abstain))
    if any(a.ndim != 1 or a.shape != y.shape for a in (label, y, abstain)):
        raise ValueError("labels, targets and abstention mask must align")
    for a in (label, y):
        numeric = np.issubdtype(a.dtype, np.number) or np.issubdtype(a.dtype, np.bool_)
        if not numeric or not np.isfinite(a).all() or np.any(a != np.floor(a)):
            raise ValueError("labels and targets must be finite integer class indices")
    if not np.isin(abstain, [0, 1]).all():
        raise ValueError("abstain must be a boolean mask")
    abstain = abstain.astype(bool)
    if np.any(y < 0) or np.any(label[~abstain] < 0):
        raise ValueError("targets and answered labels must be nonnegative class indices")
    label, y = label.astype(int), y.astype(int)
    answered = ~abstain
    n_ans = int(answered.sum())
    sel_acc = float((label[answered] == y[answered]).mean()) if n_ans else float("nan")
    return {
        "n": int(y.size),
        "coverage": float(answered.mean()) if y.size else float("nan"),
        "abstention_rate": float(abstain.mean()) if y.size else float("nan"),
        "selective_accuracy": sel_acc,
        "selective_risk": (1.0 - sel_acc) if n_ans else float("nan"),
    }
