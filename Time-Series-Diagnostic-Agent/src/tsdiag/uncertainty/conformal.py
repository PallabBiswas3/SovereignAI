"""Split-conformal prediction utilities (numpy only).

Shared by every diagnostic domain:

* classification (bearing, transformer): prediction sets with a coverage
  guarantee; a singleton set is an answer, anything else is an abstention.
* regression (turbofan RUL, battery capacity): intervals via absolute-residual
  or conformalized quantile regression (CQR).

Guarantee (marginal, finite-sample): if calibration and test points are
exchangeable, P(y in set) >= 1 - alpha.  Exchangeability must hold at the
level of the leakage unit (engine, specimen, cell, event), so calibrate on
units that are disjoint from training and from the frozen test set.
The model, temperature and scoring rule must be fixed before this calibration.
Singleton decisions do not themselves guarantee conditional selective accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np

__all__ = [
    "conformal_quantile",
    "SelectiveDecision",
    "ConformalClassifier",
    "SplitConformalRegressor",
    "ConformalizedQuantileRegressor",
    "set_coverage",
    "interval_coverage",
    "mean_interval_width",
]


def conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample conformal quantile: the ceil((n+1)(1-alpha))-th smallest score.

    Returns +inf when there are too few scores for the requested alpha
    (n < 1/alpha - 1).  Callers must treat +inf as "no usable threshold".
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    s = np.asarray(scores, dtype=float).ravel()
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite; handle unscorable units explicitly")
    n = s.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.partition(s, k - 1)[k - 1])


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def _probabilities(probs) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[1] < 1:
        raise ValueError("probs must have shape (n_samples, n_classes) with at least one class")
    if (not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8)):
        raise ValueError("probs must be finite nonnegative probabilities summing to one")
    return p


def _labels(y, n: int, k: int) -> np.ndarray:
    labels = np.asarray(y)
    numeric = np.issubdtype(labels.dtype, np.number) or np.issubdtype(labels.dtype, np.bool_)
    if (labels.ndim != 1 or labels.size != n or not numeric
            or not np.isfinite(labels).all() or np.any(labels != np.floor(labels))
            or np.any(labels < 0) or np.any(labels >= k)):
        raise ValueError("y must contain aligned integer class indices in [0, K)")
    return labels.astype(int)


def _aligned(*values, allow_empty: bool = False) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value, dtype=float) for value in values)
    if (any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays)
            or any(not np.isfinite(a).all() for a in arrays)
            or (not allow_empty and not arrays[0].size)):
        raise ValueError("inputs must be aligned finite one-dimensional arrays")
    return arrays


def _scale(scale, reference):
    s = np.ones_like(reference) if scale is None else np.asarray(scale, dtype=float)
    if s.ndim == 0:
        s = np.full_like(reference, s)
    if s.shape != reference.shape or not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError("scale must be positive, finite and aligned with predictions")
    return s


def _clip_interval(lo, hi, clip_min):
    if clip_min is not None:
        if not np.isfinite(clip_min):
            raise ValueError("clip_min must be finite")
        # Appropriate only when the target is known to have this lower support.
        lo, hi = np.maximum(lo, clip_min), np.maximum(hi, clip_min)
    return lo, hi


def _score_matrix(probs: np.ndarray, method: str) -> np.ndarray:
    """Conformity score of every candidate label (lower = more conforming)."""
    p = _probabilities(probs)
    if method == "lac":
        return 1.0 - p
    if method == "aps":
        order = np.argsort(-p, axis=1, kind="stable")
        cum = np.cumsum(np.take_along_axis(p, order, axis=1), axis=1)
        out = np.empty_like(cum)
        np.put_along_axis(out, order, cum, axis=1)
        return out
    raise ValueError(f"unknown method {method!r}; use 'lac' or 'aps'")


@dataclass(frozen=True)
class SelectiveDecision:
    """Result of turning prediction sets into answer-or-abstain decisions."""

    label: np.ndarray       # predicted class, -1 where abstaining
    abstain: np.ndarray     # bool
    set_size: np.ndarray    # int
    sets: np.ndarray        # bool (n, K)

    @property
    def coverage(self) -> float:
        """Fraction of samples answered (not the conformal coverage)."""
        return float(1.0 - self.abstain.mean()) if self.abstain.size else float("nan")


@dataclass
class ConformalClassifier:
    """Split-conformal classifier on top of any probabilistic model.

    method: 'lac' (score = 1 - p_true) or 'aps' (adaptive sets,
            score = cumulative probability mass up to the true class).
    class_conditional: calibrate one threshold per class (coverage holds per
            class; needs enough calibration samples of every class, otherwise
            that class's threshold is +inf and it is always included).
    """

    alpha: float = 0.1
    method: Literal["lac", "aps"] = "lac"
    class_conditional: bool = False
    qhat_: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    n_calibration_: int = field(default=0, init=False)

    def fit(self, probs, y) -> "ConformalClassifier":
        probs = _probabilities(probs)
        if not probs.shape[0]:
            raise ValueError("classification calibration must not be empty")
        n_classes = probs.shape[1]
        y = _labels(y, len(probs), n_classes)
        scores = _score_matrix(probs, self.method)[np.arange(y.size), y]
        if self.class_conditional:
            q = np.array(
                [conformal_quantile(scores[y == k], self.alpha) for k in range(n_classes)]
            )
        else:
            q = np.full(n_classes, conformal_quantile(scores, self.alpha))
        self.qhat_ = q
        self.n_calibration_ = int(y.size)
        return self

    def predict_sets(self, probs) -> np.ndarray:
        if self.qhat_ is None:
            raise RuntimeError("call fit() first")
        if _probabilities(probs).shape[1] != self.qhat_.size:
            raise ValueError("prediction classes must match calibration classes")
        return _score_matrix(probs, self.method) <= self.qhat_[None, :]

    def decide(self, probs) -> SelectiveDecision:
        """Answer only when the prediction set is a single class."""
        sets = self.predict_sets(probs)
        size = sets.sum(axis=1)
        answered = size == 1
        label = np.where(answered, np.argmax(sets, axis=1), -1)
        return SelectiveDecision(label=label, abstain=~answered, set_size=size, sets=sets)


def set_coverage(sets: np.ndarray, y) -> float:
    """Fraction of samples whose true label lies in the prediction set."""
    sets = np.asarray(sets, dtype=bool)
    if sets.ndim != 2:
        raise ValueError("sets must be a two-dimensional mask")
    y = _labels(y, len(sets), sets.shape[1])
    if not y.size:
        return float("nan")
    return float(np.mean(sets[np.arange(y.size), y]))


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
class SplitConformalRegressor:
    """Symmetric intervals from absolute residuals, optionally scale-normalised.

    Pass `scale` (e.g. an ensemble std, or predicted RUL-dependent noise) at
    both fit and predict time for heteroscedastic intervals.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.qhat_: Optional[float] = None

    def fit(self, y, y_pred, scale=None) -> "SplitConformalRegressor":
        y, y_pred = _aligned(y, y_pred)
        s = _scale(scale, y)
        self.qhat_ = conformal_quantile(np.abs(y - y_pred) / s, self.alpha)
        return self

    def predict_interval(
        self, y_pred, scale=None, clip_min: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.qhat_ is None:
            raise RuntimeError("call fit() first")
        y_pred, = _aligned(y_pred, allow_empty=True)
        s = _scale(scale, y_pred)
        lo, hi = y_pred - self.qhat_ * s, y_pred + self.qhat_ * s
        return _clip_interval(lo, hi, clip_min)


class ConformalizedQuantileRegressor:
    """CQR (Romano et al. 2019): widen/narrow model quantile bands to be valid."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.qhat_: Optional[float] = None

    def fit(self, y, lo, hi) -> "ConformalizedQuantileRegressor":
        y, lo, hi = _aligned(y, lo, hi)
        if np.any(lo > hi):
            raise ValueError("model lower quantiles must not exceed upper quantiles")
        self.qhat_ = conformal_quantile(np.maximum(lo - y, y - hi), self.alpha)
        return self

    def predict_interval(
        self, lo, hi, clip_min: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.qhat_ is None:
            raise RuntimeError("call fit() first")
        lo, hi = _aligned(lo, hi, allow_empty=True)
        if np.any(lo > hi):
            raise ValueError("model lower quantiles must not exceed upper quantiles")
        return _clip_interval(lo - self.qhat_, hi + self.qhat_, clip_min)


def interval_coverage(y, lo, hi) -> float:
    # Infinite endpoints represent insufficient calibration; do not silently drop them.
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    if any(a.ndim != 1 or a.shape != y.shape for a in (y, lo, hi)):
        raise ValueError("targets and intervals must be aligned one-dimensional arrays")
    if not np.isfinite(y).all() or np.isnan(lo).any() or np.isnan(hi).any():
        raise ValueError("targets must be finite and intervals must not contain NaN")
    if not y.size:
        return float("nan")
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_interval_width(lo, hi) -> float:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if lo.ndim != 1 or lo.shape != hi.shape or np.isnan(lo).any() or np.isnan(hi).any():
        raise ValueError("intervals must be aligned one-dimensional arrays without NaN")
    # CQR can shrink a band into an empty set; its width is zero, not negative.
    return float(np.maximum(hi - lo, 0).mean()) if lo.size else float("nan")
