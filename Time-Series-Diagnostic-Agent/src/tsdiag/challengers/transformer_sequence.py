"""Symmetrical-component features and an imbalance-aware classifier for
transformer waveforms (SGAH-style 100 x 6 windows: Ua, Ub, Uc, Ia, Ib, Ic).

Why
---
Inter-phase short circuits (IPSF) were never recalled although main-transformer
fault *detection* is perfect.  A deep model fed raw waveforms has to discover
that these faults live in the NEGATIVE-sequence and line-to-line quantities;
the Fortescue transform exposes them directly:

    balanced load      -> V1, I1 only
    phase-to-phase     -> large negative-sequence I2 (and V2), little zero-sequence
    phase-to-ground    -> large zero-sequence I0 / V0
    energisation/inrush-> large 2nd-harmonic share of the current

Sequence/harmonic ratios describe measured imbalance. Current/voltage coupling
features retain unit and load dependence; transfer must be evaluated separately.
Zero-sequence observability and named channel order must be verified from the
measurement setup before interpreting these features as physical fault evidence.

Assumption to check against the dataset description: how many fundamental
cycles the 100-sample window spans.  Pass `cycles_in_window` if known; otherwise
the dominant spectral bin of the voltages is used.

Protocol rules: fit scalers/classifiers on training records only, choose any
hyper-parameters on dev data, and score the frozen test set once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..calibration.event_conformal import wilson_interval

__all__ = [
    "SequenceFeatureExtractor",
    "augment_waveforms",
    "oversample_minority",
    "SequenceFeatureClassifier",
    "per_class_recall",
]

_A = np.exp(2j * np.pi / 3.0)
_EPS = 1e-9


def _waveforms(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[None]
    if (X.ndim != 3 or X.shape[2] != 6 or not X.shape[0] or X.shape[1] < 8
            or not np.isfinite(X).all()):
        raise ValueError("expected finite nonempty (N, T>=8, 6) waveforms = Ua,Ub,Uc,Ia,Ib,Ic")
    return X


def _labels(y, n):
    y = np.asarray(y)
    if (y.ndim != 1 or len(y) != n or not np.issubdtype(y.dtype, np.number)
            or not np.isfinite(y).all() or np.any(y != np.floor(y))):
        raise ValueError("labels must be aligned integer class IDs")
    return y.astype(int)


def _phasor(x: np.ndarray, k) -> np.ndarray:
    """Peak-amplitude phasor of bin k for x of shape (N, T, ...) along axis 1."""
    T = x.shape[1]
    bins = np.broadcast_to(np.asarray(k, float), (len(x),))
    phase = 2 * np.pi * bins[:, None] * np.arange(T)[None] / T
    if np.all(bins == np.floor(bins)):
        return (2.0 / T) * np.einsum("ntc,nt->nc", x, np.exp(-1j * phase))
    # Non-integer cycle metadata must not be rounded to a different frequency.
    out = np.empty((len(x), x.shape[2]), complex)
    for row in range(len(x)):
        design = np.column_stack([np.cos(phase[row]), np.sin(phase[row]), np.ones(T)])
        coefficients = np.linalg.lstsq(design, x[row], rcond=None)[0]
        out[row] = coefficients[0] - 1j * coefficients[1]
    return out


def _sequence(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    z = (a + b + c) / 3.0
    p = (a + _A * b + _A**2 * c) / 3.0
    n = (a + _A**2 * b + _A * c) / 3.0
    return z, p, n


@dataclass
class SequenceFeatureExtractor:
    """(N, T, 6) waveforms -> (N, F) scale-free sequence/harmonic features."""

    cycles_in_window: Optional[float] = None
    n_harmonics: int = 7
    feature_names_: List[str] = field(default_factory=list, init=False)

    def _bin(self, X: np.ndarray) -> np.ndarray:
        if self.cycles_in_window is not None:
            if not np.isfinite(self.cycles_in_window) or not 0 < self.cycles_in_window < X.shape[1] / 2:
                raise ValueError("cycles_in_window must be positive and below Nyquist")
            return np.full(len(X), float(self.cycles_in_window))
        v = X[:, :, :3] - X[:, :, :3].mean(1, keepdims=True)
        spec = np.abs(np.fft.rfft(v, axis=1)).sum(axis=2)
        usable = spec[:, 1 : X.shape[1] // 4]
        if np.any(usable.max(axis=1) <= _EPS):
            raise ValueError("cannot estimate a frequency from constant/zero voltages; supply cycle metadata")
        return np.argmax(usable, axis=1) + 1

    def transform(self, X) -> np.ndarray:
        X = _waveforms(X)
        if not isinstance(self.n_harmonics, (int, np.integer)) or self.n_harmonics < 1:
            raise ValueError("n_harmonics must be a positive integer")
        k = self._bin(X)
        T = X.shape[1]
        feats: Dict[str, np.ndarray] = {}

        for name, sl in (("V", slice(0, 3)), ("I", slice(3, 6))):
            w = X[:, :, sl]
            ph = _phasor(w, k)                                  # (N, 3)
            z, p, n = _sequence(ph[:, 0], ph[:, 1], ph[:, 2])
            p_mag = np.abs(p) + _EPS
            feats[f"{name}_neg_over_pos"] = np.abs(n) / p_mag
            feats[f"{name}_zero_over_pos"] = np.abs(z) / p_mag
            feats[f"{name}_neg_phase_vs_pos"] = np.cos(np.angle(n) - np.angle(p))
            feats[f"{name}_zero_phase_vs_pos"] = np.cos(np.angle(z) - np.angle(p))
            mags = np.abs(ph)
            feats[f"{name}_mag_max_over_mean"] = mags.max(1) / (mags.mean(1) + _EPS)
            feats[f"{name}_mag_min_over_mean"] = mags.min(1) / (mags.mean(1) + _EPS)
            # time-domain zero sequence (independent of phasor estimation)
            rms = np.sqrt((w**2).mean(axis=(1, 2))) + _EPS
            feats[f"{name}_td_zero_rms"] = np.sqrt(((w.sum(2) / 3.0) ** 2).mean(1)) / rms
            # line-to-line content relative to phase content
            ll = np.stack([w[:, :, 0] - w[:, :, 1], w[:, :, 1] - w[:, :, 2], w[:, :, 2] - w[:, :, 0]], 2)
            feats[f"{name}_ll_rms_ratio"] = np.sqrt((ll**2).mean(axis=(1, 2))) / (np.sqrt(3.0) * rms)
            feats[f"{name}_ll_rms_spread"] = np.std(np.sqrt((ll**2).mean(1)), axis=1) / (
                np.mean(np.sqrt((ll**2).mean(1)), axis=1) + _EPS
            )
            # harmonic content of the three phases (shares of the fundamental)
            fund = np.abs(ph).mean(1) + _EPS
            for h in (2, 3, 5):
                observable = (h * k < T / 2) & (h <= self.n_harmonics)
                share = np.zeros(len(X))
                if observable.any():
                    share[observable] = np.abs(_phasor(w[observable], h * k[observable])).mean(1) / fund[observable]
                feats[f"{name}_h{h}_share"] = share
                feats[f"{name}_h{h}_observable"] = observable.astype(float)
            harmonic_energy = np.zeros(len(X))
            harmonic_count = np.zeros(len(X))
            for h in range(2, self.n_harmonics + 1):
                observable = h * k < T / 2
                if observable.any():
                    share = np.abs(_phasor(w[observable], h * k[observable])).mean(1) / fund[observable]
                    harmonic_energy[observable] += share**2
                    harmonic_count[observable] += 1
            feats[f"{name}_thd"] = np.sqrt(harmonic_energy)
            feats[f"{name}_thd_harmonic_count"] = harmonic_count

        # current-voltage coupling (fault direction / power-factor angle)
        v_ph, i_ph = _phasor(X[:, :, :3], k), _phasor(X[:, :, 3:], k)
        _, v1, v2 = _sequence(v_ph[:, 0], v_ph[:, 1], v_ph[:, 2])
        _, i1, i2 = _sequence(i_ph[:, 0], i_ph[:, 1], i_ph[:, 2])
        feats["pf_angle_cos"] = np.cos(np.angle(v1) - np.angle(i1))
        feats["pf_angle_sin"] = np.sin(np.angle(v1) - np.angle(i1))
        feats["neg_seq_direction_cos"] = np.cos(np.angle(v2) - np.angle(i2))
        feats["i_pos_over_v_pos"] = np.abs(i1) / (np.abs(v1) + _EPS)
        # overall level of current relative to voltage for the whole window
        feats["i_rms_over_v_rms"] = np.sqrt((X[:, :, 3:] ** 2).mean(axis=(1, 2))) / (
            np.sqrt((X[:, :, :3] ** 2).mean(axis=(1, 2))) + _EPS
        )
        feats["crest_i"] = np.abs(X[:, :, 3:]).max(axis=(1, 2)) / (
            np.sqrt((X[:, :, 3:] ** 2).mean(axis=(1, 2))) + _EPS
        )

        self.feature_names_ = list(feats)
        return np.column_stack([feats[n] for n in self.feature_names_])


# --------------------------------------------------------------------------- #
# Imbalance handling
# --------------------------------------------------------------------------- #
def augment_waveforms(
    X: np.ndarray,
    rng: np.random.Generator,
    noise: float = 0.02,
    max_shift: Optional[int] = None,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    whole_cycles: bool = False,
) -> np.ndarray:
    """Label-preserving augmentations for ~periodic windows.

    * additive Gaussian noise (relative to each record's RMS)
    * circular time shift (valid only when the window spans whole cycles)
    * global amplitude scaling (the features are scale-free ratios)
    """
    X = _waveforms(X)
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    if len(scale_range) != 2 or not np.isfinite(scale_range).all() or not 0 < scale_range[0] <= scale_range[1]:
        raise ValueError("scale_range must contain finite positive ordered bounds")
    out = X.copy()
    N, T, _ = X.shape
    rms = np.sqrt((X**2).mean(axis=1, keepdims=True)) + _EPS
    out += rng.normal(size=X.shape) * noise * rms
    if max_shift is None:
        max_shift = T // 10 if whole_cycles else 0
    if not isinstance(max_shift, (int, np.integer)) or max_shift < 0:
        raise ValueError("max_shift must be a nonnegative integer")
    if max_shift and not whole_cycles:
        raise ValueError("circular shifts require verified whole-cycle windows")
    shifts = rng.integers(-max_shift, max_shift + 1, size=N)
    for i, s in enumerate(shifts):
        out[i] = np.roll(out[i], int(s), axis=0)
    out *= rng.uniform(*scale_range, size=(N, 1, 1))
    return out


def oversample_minority(
    X: np.ndarray, y: np.ndarray, min_per_class: int, rng: np.random.Generator, **aug
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment classes below `min_per_class` until they reach it (training data only)."""
    X = _waveforms(X)
    y = _labels(y, len(X))
    if not isinstance(min_per_class, (int, np.integer)) or min_per_class < 1:
        raise ValueError("min_per_class must be a positive integer")
    Xs, ys = [X], [y]
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        need = min_per_class - idx.size
        if need > 0:
            pick = rng.choice(idx, size=need, replace=True)
            Xs.append(augment_waveforms(X[pick], rng, **aug))
            ys.append(np.full(need, c))
    return np.concatenate(Xs), np.concatenate(ys)


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
class SequenceFeatureClassifier:
    """Class-weighted gradient boosting on sequence features.

    Can be stacked with an existing model: pass its class probabilities as
    `extra` (columns appended to the features) at both fit and predict time.
    Training stacking probabilities must be out-of-fold or supplied by an
    independently frozen base model. Probability columns follow classes_.
    """

    def __init__(
        self,
        extractor: Optional[SequenceFeatureExtractor] = None,
        min_per_class: int = 200,
        augment: bool = True,
        seed: int = 0,
    ):
        self.extractor = extractor or SequenceFeatureExtractor()
        self.min_per_class = min_per_class
        self.augment = augment
        self.seed = seed
        self.clf_ = None
        self._fitted_extractor = None
        self._extra_columns = None

    def _features(self, X, extra=None):
        extractor = self._fitted_extractor or self.extractor
        F = extractor.transform(X)
        if extra is None:
            if self._extra_columns is not None:
                raise ValueError("prediction requires the same extra probability columns used at fit")
            return F
        extra = np.asarray(extra, float)
        if (extra.ndim != 2 or len(extra) != len(F) or not extra.shape[1]
                or not np.isfinite(extra).all() or np.any(extra < 0)
                or not np.allclose(extra.sum(axis=1), 1.0)
                or (self._extra_columns is not None and extra.shape[1] != self._extra_columns)):
            raise ValueError("extra must be aligned class probabilities with the fitted column count")
        return np.hstack([F, extra])

    def fit(self, X, y, extra=None) -> "SequenceFeatureClassifier":
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.clf_, self._fitted_extractor, self._extra_columns = None, None, None
        X = _waveforms(X)
        y = _labels(y, len(X))
        if len(np.unique(y)) < 2:
            raise ValueError("classification training needs at least two classes")
        rng = np.random.default_rng(self.seed)
        if self.augment:
            # extra (model probabilities) cannot be recomputed for synthetic windows,
            # so augmentation is only used for the pure-waveform variant.
            if extra is not None:
                raise ValueError("augment=True is not supported together with `extra`")
            whole = (self.extractor.cycles_in_window is not None
                     and float(self.extractor.cycles_in_window).is_integer())
            X, y = oversample_minority(X, y, self.min_per_class, rng, whole_cycles=whole)
        self.clf_ = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=16,
            l2_regularization=1.0, class_weight="balanced", random_state=self.seed,
        ).fit(self._features(X, extra), y)
        self._fitted_extractor = deepcopy(self.extractor)
        self._extra_columns = None if extra is None else np.asarray(extra).shape[1]
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X, extra=None) -> np.ndarray:
        if self.clf_ is None:
            raise RuntimeError("call fit() before predicting")
        if (extra is not None) != (self._extra_columns is not None):
            raise ValueError("extra probability columns must match the fitted feature schema")
        return self.clf_.predict_proba(self._features(X, extra))

    def predict(self, X, extra=None) -> np.ndarray:
        probabilities = self.predict_proba(X, extra)
        return self.classes_[probabilities.argmax(1)]


def per_class_recall(y_true, y_pred) -> Dict[int, Dict[str, object]]:
    """Recall per class with a Wilson interval (small classes need the interval)."""
    y_true = _labels(y_true, len(y_true))
    y_pred = _labels(y_pred, len(y_true))
    out = {}
    for c in np.unique(y_true):
        m = y_true == c
        k, n = int((y_pred[m] == c).sum()), int(m.sum())
        out[int(c)] = {"recall": k / n, "n": n, "wilson95": wilson_interval(k, n)}
    return out
