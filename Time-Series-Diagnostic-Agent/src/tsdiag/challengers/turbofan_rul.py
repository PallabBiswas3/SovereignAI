"""Turbofan (C-MAPSS) RUL challenger with calibrated intervals.

Design
------
* Target: piecewise-linear RUL, min(RUL, cap).  Early-life RUL is not
  observable from sensors, so training on the raw target mostly adds noise.
* Regime handling: operating settings are clustered (k=1 for FD001/FD003,
  k=6 for FD002/FD004) and every sensor is standardised *within its regime*.
* Windows: the last `window` cycles of each engine (front-padded by repeating
  the first row for short histories).
* Backends (same interface):
    "hgb"  gradient boosting on window summary features (last value, mean,
           std, slope).  Strong baseline, no deep-learning dependency.
    "cnn"  1D-CNN in PyTorch (optional dependency, imported lazily).
  Each backend is trained as a bootstrap-over-engines ensemble; the ensemble
  standard deviation is the per-sample scale for the conformal interval.
* Intervals: split-conformal on |y - mean| / (std + floor), calibrated on
  engines disjoint from training.  The model is trained on engines A,
  calibrated on engines B, and the frozen test engines are used once.

Protocol rules
--------------
* Split by ENGINE, never by row/window.
* Calibration should mimic test time: one truncation point per identified engine
  (`truncated_samples`). Repeated cuts from one engine are dependent and are
  rejected by conformal calibration and engine-level evaluation.
* Point-model training uses capped labels. Final interval calibration defaults
  to uncapped RUL; optional capped-target intervals are explicitly identified.
* Report RMSE and the NASA asymmetric score on the last window of each test
  engine, against the *uncapped* true RUL supplied with the benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from ..uncertainty.conformal import SplitConformalRegressor, interval_coverage, mean_interval_width

__all__ = [
    "cap_rul",
    "RegimeNormalizer",
    "make_windows",
    "truncated_samples",
    "rmse",
    "nasa_score",
    "RULChallenger",
    "evaluate_rul",
]


# --------------------------------------------------------------------------- #
# Targets and metrics
# --------------------------------------------------------------------------- #
def _positive_int(value, name):
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _vector(value, name, n=None, nonnegative=False):
    a = np.asarray(value, dtype=float)
    if (a.ndim != 1 or not len(a) or not np.isfinite(a).all()
            or (n is not None and len(a) != n) or (nonnegative and np.any(a < 0))):
        raise ValueError(f"{name} must be a finite nonempty aligned vector")
    return a


def _matrix(value, name, n=None):
    a = np.asarray(value, dtype=float)
    if (a.ndim != 2 or not a.shape[0] or not a.shape[1] or not np.isfinite(a).all()
            or (n is not None and len(a) != n)):
        raise ValueError(f"{name} must be a finite nonempty aligned matrix")
    return a


def _unit_keys(units, n, unique=False):
    if units is None:
        raise ValueError("engine identities are required")
    a = np.asarray(units)
    if a.ndim != 1 or len(a) != n:
        raise ValueError("engine identities must align with samples")
    if np.issubdtype(a.dtype, np.number):
        if not np.isfinite(a).all() or np.any(a != np.floor(a)):
            raise ValueError("numeric engine identities must be finite integers")
        keys = np.asarray([str(int(v)) for v in a])
    elif a.dtype.kind in {"U", "S", "O"}:
        if any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in a):
            raise ValueError("engine identities must be nonempty strings or integers")
        keys = a.astype(str)
    else:
        raise ValueError("engine identities must be strings or integers")
    if unique and len(np.unique(keys)) != n:
        raise ValueError("use exactly one window per independent engine")
    return keys


def _rows(units, cycles, x):
    x = _matrix(x, "sensor rows")
    cycles = _vector(cycles, "cycles", len(x))
    if np.any(cycles <= 0) or np.any(cycles != np.floor(cycles)):
        raise ValueError("cycles must be positive integer indices")
    _unit_keys(units, len(x))
    units = np.asarray(units)
    for unit in np.unique(units):
        values = cycles[units == unit]
        if len(np.unique(values)) != len(values):
            raise ValueError("duplicate cycle within an engine")
    return units, cycles, x


def _windows(windows, window=None, n_features=None):
    a = np.asarray(windows, dtype=float)
    if (a.ndim != 3 or not all(a.shape) or not np.isfinite(a).all()
            or (window is not None and a.shape[1] != window)
            or (n_features is not None and a.shape[2] != n_features)):
        raise ValueError("windows must be finite nonempty (N, W, F) with the fitted shape")
    return a


def cap_rul(rul, cap: float = 125.0) -> np.ndarray:
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("cap must be positive and finite")
    return np.minimum(_vector(rul, "RUL", nonnegative=True), cap)


def rmse(y_true, y_pred) -> float:
    y_true = _vector(y_true, "true RUL", nonnegative=True)
    d = _vector(y_pred, "predicted RUL", len(y_true)) - y_true
    return float(np.sqrt(np.mean(d**2)))


def nasa_score(y_true, y_pred) -> float:
    """C-MAPSS asymmetric score: late predictions (pred > true) cost more."""
    y_true = _vector(y_true, "true RUL", nonnegative=True)
    d = _vector(y_pred, "predicted RUL", len(y_true)) - y_true
    penalty = np.where(d < 0, -d / 13.0, d / 10.0)
    return float(np.expm1(penalty).sum())


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
class RegimeNormalizer:
    """Cluster operating settings and standardise sensors within each regime."""

    def __init__(self, n_regimes: int = 1, min_std: float = 1e-6, seed: int = 0):
        _positive_int(n_regimes, "n_regimes")
        if not np.isfinite(min_std) or min_std <= 0:
            raise ValueError("min_std must be positive and finite")
        self.n_regimes = n_regimes
        self.min_std = min_std
        self.seed = seed
        self._fitted = False

    def fit(self, settings: np.ndarray, sensors: np.ndarray) -> "RegimeNormalizer":
        self._fitted = False
        settings = _matrix(settings, "operating settings")
        sensors = _matrix(sensors, "sensors", len(settings))
        if len(np.unique(settings, axis=0)) < self.n_regimes:
            raise ValueError("not enough distinct operating settings for n_regimes")
        self.settings_mean_ = settings.mean(axis=0)
        self.settings_std_ = np.maximum(settings.std(axis=0), self.min_std)
        scaled_settings = (settings - self.settings_mean_) / self.settings_std_
        if self.n_regimes > 1:
            from sklearn.cluster import KMeans

            self.km_ = KMeans(self.n_regimes, n_init=10, random_state=self.seed).fit(scaled_settings)
            reg = self.km_.labels_
        else:
            self.km_, reg = None, np.zeros(len(sensors), dtype=int)
        self.mean_ = np.zeros((self.n_regimes, sensors.shape[1]))
        self.std_ = np.zeros((self.n_regimes, sensors.shape[1]))
        for r in range(self.n_regimes):
            m = reg == r
            if m.any():
                self.mean_[r] = sensors[m].mean(0)
                self.std_[r] = sensors[m].std(0)
        # sensors that are (near-)constant in every regime carry no information
        self.keep_ = (self.std_ > self.min_std).any(axis=0)
        if not self.keep_.any():
            raise ValueError("no variable sensors remain after regime normalization")
        self.std_ = np.where(self.std_ > self.min_std, self.std_, 1.0)
        self._fitted = True
        return self

    def regime(self, settings: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before assigning regimes")
        settings = _matrix(settings, "operating settings")
        if settings.shape[1] != len(self.settings_mean_):
            raise ValueError("operating setting columns differ from fit")
        if self.km_ is None:
            return np.zeros(len(settings), dtype=int)
        return self.km_.predict((settings - self.settings_mean_) / self.settings_std_)

    def transform(self, settings: np.ndarray, sensors: np.ndarray) -> np.ndarray:
        r = self.regime(settings)
        sensors = _matrix(sensors, "sensors", len(r))
        if sensors.shape[1] != self.mean_.shape[1]:
            raise ValueError("sensor columns differ from fit")
        z = (sensors - self.mean_[r]) / self.std_[r]
        return z[:, self.keep_]


def make_windows(
    units: np.ndarray,
    cycles: np.ndarray,
    x: np.ndarray,
    window: int = 30,
    last_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-engine sliding windows ending at each cycle (or only the last one).

    Returns (windows (N, W, F), unit_ids (N,), cycle_idx (N,)).  Rows must be
    sorted by (unit, cycle); short histories are front-padded with the first row.
    """
    _positive_int(window, "window")
    units, cycles, x = _rows(units, cycles, x)
    out, uid, cyc = [], [], []
    for u in np.unique(units):
        idx = np.where(units == u)[0]
        order = np.argsort(cycles[idx], kind="stable")
        xi, ci = x[idx][order], cycles[idx][order]
        pad = np.vstack([np.repeat(xi[:1], window - 1, axis=0), xi])
        ends = [len(xi) - 1] if last_only else range(len(xi))
        for e in ends:
            out.append(pad[e : e + window])
            uid.append(u)
            cyc.append(ci[e])
    return np.stack(out), np.asarray(uid), np.asarray(cyc)


def truncated_samples(
    units: np.ndarray,
    cycles: np.ndarray,
    x: np.ndarray,
    rul: np.ndarray,
    window: int = 30,
    cuts_per_engine: int = 1,
    min_cycles: int = 30,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mimic the benchmark's random truncation: windows ending at random cycles.

    `rul` is the per-row (uncapped) remaining life.  Returns (windows, rul, unit_ids).
    """
    rng = np.random.default_rng(seed)
    _positive_int(window, "window")
    _positive_int(cuts_per_engine, "cuts_per_engine")
    _positive_int(min_cycles, "min_cycles")
    units, cycles, x = _rows(units, cycles, x)
    rul = _vector(rul, "RUL", len(x), nonnegative=True)
    W, R, U = [], [], []
    for u in np.unique(units):
        idx = np.where(units == u)[0]
        order = np.argsort(cycles[idx], kind="stable")
        xi, ri = np.asarray(x)[idx][order], np.asarray(rul)[idx][order]
        lo = min(min_cycles, len(xi)) - 1
        cuts = rng.integers(lo, len(xi), size=cuts_per_engine)
        pad = np.vstack([np.repeat(xi[:1], window - 1, axis=0), xi])
        for c in cuts:
            W.append(pad[c : c + window])
            R.append(ri[c])
            U.append(u)
    return np.stack(W), np.asarray(R, float), np.asarray(U)


def window_features(windows: np.ndarray) -> np.ndarray:
    """Summary features per window: last value, mean, std, linear slope per sensor."""
    windows = _windows(windows)
    n, w, f = windows.shape
    t = np.arange(w) - (w - 1) / 2.0
    slope = (windows * t[None, :, None]).sum(1) / (t**2).sum() if w > 1 else np.zeros((n, f))
    return np.hstack([windows[:, -1], windows.mean(1), windows.std(1), slope])


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class _HGBMember:
    def __init__(self, seed: int):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=24,
            l2_regularization=1.0, random_state=seed,
        )

    def fit(self, windows, y):
        self.m.fit(window_features(windows), y)
        return self

    def predict(self, windows):
        return self.m.predict(window_features(windows))


class _CNNMember:
    """1D-CNN regressor (requires PyTorch)."""

    def __init__(self, seed: int, epochs: int = 40, lr: float = 2e-3, batch: int = 256, cap: float = 125.0):
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("backend='cnn' needs PyTorch (pip install torch)") from exc
        self.seed, self.epochs, self.lr, self.batch, self.cap = seed, epochs, lr, batch, cap

    def _build(self, n_feat: int):
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv1d(n_feat, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def fit(self, windows, y):
        import torch

        torch.manual_seed(self.seed)
        X = torch.tensor(windows, dtype=torch.float32).transpose(1, 2)   # (N, F, W)
        t = torch.tensor(y / self.cap, dtype=torch.float32)
        self.net = self._build(X.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        g = torch.Generator().manual_seed(self.seed)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(X), generator=g)
            for i in range(0, len(X), self.batch):
                b = perm[i : i + self.batch]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(self.net(X[b]).squeeze(1), t[b])
                loss.backward()
                opt.step()
        return self

    def predict(self, windows):
        import torch

        self.net.eval()
        with torch.no_grad():
            X = torch.tensor(windows, dtype=torch.float32).transpose(1, 2)
            return self.net(X).squeeze(1).numpy() * self.cap


# --------------------------------------------------------------------------- #
# Challenger
# --------------------------------------------------------------------------- #
@dataclass
class RULChallenger:
    """Bootstrap-ensemble RUL regressor with conformal intervals."""

    backend: Literal["hgb", "cnn"] = "hgb"
    n_members: int = 5
    cap: float = 125.0
    window: int = 30
    alpha: float = 0.1
    scale_floor: float = 1.0          # RUL units; keeps intervals non-degenerate
    seed: int = 0
    members_: List = field(default_factory=list, init=False, repr=False)
    conformal_: Optional[SplitConformalRegressor] = field(default=None, init=False, repr=False)
    training_units_: frozenset[str] = field(default_factory=frozenset, init=False)
    calibration_units_: frozenset[str] = field(default_factory=frozenset, init=False)
    interval_target_: Optional[str] = field(default=None, init=False)
    _fitted_config: Optional[tuple] = field(default=None, init=False, repr=False)
    _n_features: Optional[int] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._validate_config()

    def _validate_config(self):
        if self.backend not in {"hgb", "cnn"}:
            raise ValueError("backend must be hgb or cnn")
        _positive_int(self.n_members, "n_members")
        _positive_int(self.window, "window")
        if not np.isfinite(self.cap) or self.cap <= 0:
            raise ValueError("cap must be positive and finite")
        if not np.isfinite(self.scale_floor) or self.scale_floor <= 0:
            raise ValueError("scale_floor must be positive and finite")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")

    def _config(self):
        return (self.backend, self.n_members, self.cap, self.window,
                self.alpha, self.scale_floor, self.seed)

    def _check_fitted(self):
        if not self.members_ or self._fitted_config is None:
            raise RuntimeError("call fit() before predicting")
        if self._config() != self._fitted_config:
            raise ValueError("model configuration changed; refit before using predictions or intervals")

    def _member(self, seed: int):
        return _HGBMember(seed) if self.backend == "hgb" else _CNNMember(seed, cap=self.cap)

    def fit(self, windows: np.ndarray, rul: np.ndarray, units: np.ndarray) -> "RULChallenger":
        """Train on windows with (uncapped) RUL labels; bootstrap resamples ENGINES."""
        self.members_, self.conformal_, self._fitted_config = [], None, None
        self.training_units_, self.calibration_units_ = frozenset(), frozenset()
        self.interval_target_, self._n_features = None, None
        self._validate_config()
        windows = _windows(windows, self.window)
        units = _unit_keys(units, len(windows))
        rul = _vector(rul, "training RUL", len(windows), nonnegative=True)
        y = cap_rul(rul, self.cap)
        rng = np.random.default_rng(self.seed)
        uniq = np.unique(units)
        members = []
        for m in range(self.n_members):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.where(units == u)[0] for u in pick])
            members.append(self._member(self.seed + m).fit(windows[idx], y[idx]))
        self.members_ = members
        self.training_units_ = frozenset(uniq)
        self._fitted_config, self._n_features = self._config(), windows.shape[2]
        return self

    def predict(self, windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        windows = _windows(windows, self.window, self._n_features)
        preds = np.stack([m.predict(windows) for m in self.members_])
        if preds.shape != (len(self.members_), len(windows)) or not np.isfinite(preds).all():
            raise ValueError("ensemble produced invalid or misaligned predictions")
        return np.clip(preds.mean(0), 0.0, self.cap), preds.std(0)

    def calibrate(self, windows: np.ndarray, rul: np.ndarray, units=None,
                  *, target: Literal["uncapped", "capped"] = "uncapped") -> "RULChallenger":
        """Calibrate one window per held-out engine against the explicit RUL target.

        Truncation/scoring rules must be fixed before calibration and match the
        eventual evaluation protocol. Random prefixes do not prove equivalence
        to the official C-MAPSS censoring distribution.
        """
        self.conformal_, self.calibration_units_, self.interval_target_ = None, frozenset(), None
        self._check_fitted()
        windows = _windows(windows, self.window, self._n_features)
        keys = _unit_keys(units, len(windows), unique=True)
        if set(keys) & self.training_units_:
            raise ValueError("calibration engines must be disjoint from training engines")
        if target not in {"uncapped", "capped"}:
            raise ValueError("interval target must be uncapped or capped")
        rul = _vector(rul, "calibration RUL", len(windows), nonnegative=True)
        mu, sd = self.predict(windows)
        calibration = SplitConformalRegressor(self.alpha).fit(
            rul if target == "uncapped" else cap_rul(rul, self.cap), mu,
            scale=sd + self.scale_floor
        )
        if not np.isfinite(calibration.qhat_):
            raise ValueError("too few independent calibration engines for a finite conformal interval")
        self.conformal_, self.calibration_units_, self.interval_target_ = calibration, frozenset(keys), target
        return self

    def predict_interval(self, windows: np.ndarray, units=None):
        if self.conformal_ is None:
            raise RuntimeError("call calibrate() first")
        self._check_fitted()
        windows = _windows(windows, self.window, self._n_features)
        keys = _unit_keys(units, len(windows), unique=True)
        if set(keys) & (self.training_units_ | self.calibration_units_):
            raise ValueError("interval prediction engines must be disjoint from training and calibration")
        mu, sd = self.predict(windows)
        lo, hi = self.conformal_.predict_interval(mu, scale=sd + self.scale_floor, clip_min=0.0)
        return mu, lo, hi


def evaluate_rul(model: RULChallenger, windows: np.ndarray, true_rul, units=None) -> Dict[str, object]:
    """Benchmark metrics on the last window of each test engine.

    One window per independent test engine is required. Point metrics always
    use uncapped RUL. Both interval coverages and the calibration target are
    reported explicitly; capped coverage cannot certify uncapped RUL coverage.
    """
    mu, lo, hi = model.predict_interval(windows, units=units)
    true_rul = _vector(true_rul, "true RUL", len(mu), nonnegative=True)
    capped = cap_rul(true_rul, model.cap)
    return {
        "rmse": rmse(true_rul, mu),
        "mae": float(np.abs(true_rul - mu).mean()),
        "nasa_score": nasa_score(true_rul, mu),
        "interval_target": model.interval_target_,
        "interval_coverage_uncapped": interval_coverage(true_rul, lo, hi),
        "interval_coverage_capped": interval_coverage(capped, lo, hi),
        "mean_interval_width": mean_interval_width(lo, hi),
        "n_engines": int(len(mu)),
    }
