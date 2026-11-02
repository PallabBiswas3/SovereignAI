"""Battery capacity-fade prognostics: particle filter, Gaussian process, and
censoring-aware leave-one-cell-out evaluation.

Fade model (3 parameters, linear in the limit g -> 0):

    Q(k) = q0 - s * (exp(g k) - 1) / g

q0 is the initial capacity, s the initial fade rate per cycle and g the
acceleration.  End of life (EOL) is the first cycle at which Q <= `eol_capacity`.

Two predictors share an interface and return approximate model-based RUL samples.
Their intervals are not conformally calibrated or guaranteed posterior coverage:

* FadeParticleFilter  - Bayesian filtering of (q0, log s, g) with a prior learned
  from the other cells (leave-one-cell-out means the held-out cell is never in
  the prior).  RUL per particle is the analytic threshold crossing.
* GPFadeModel         - parametric mean + Gaussian-process residual, with sampled
  fit-parameter uncertainty. No crossing by its finite horizon is right-censored,
  not evidence that the cell can never reach EOL.

Censoring: a cell that never reaches EOL (your B0007 case) has no true RUL, only
a lower bound.  It is excluded from error metrics and checked separately with a
survival-consistency score (does the model put its mass beyond what was already
observed?).

With only a handful of cells, every number is high-variance.  Report intervals,
and evaluate per cell, not just pooled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

__all__ = [
    "CellData",
    "fade_curve",
    "crossing_cycle",
    "fit_fade",
    "FadeParticleFilter",
    "GPFadeModel",
    "weighted_quantiles",
    "alpha_lambda_hits",
    "prognostic_horizon",
    "survival_consistency",
    "leave_one_cell_out",
    "time_to_voltage",
    "incremental_capacity",
    "ic_peak",
]

_G_MIN, _G_MAX = -0.01, 0.06


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _integer(value, name, minimum=1):
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _history(cycles, capacity, minimum=3):
    c, y = np.asarray(cycles, float), np.asarray(capacity, float)
    if (c.ndim != 1 or c.shape != y.shape or len(c) < minimum
            or not np.isfinite(c).all() or not np.isfinite(y).all()
            or np.any(c < 0) or np.any(np.diff(c) <= 0) or np.any(y <= 0)):
        raise ValueError("need aligned finite positive capacities and increasing nonnegative cycles")
    return c, y


def _distribution(values, weights):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    if (v.ndim != 1 or not len(v) or v.shape != w.shape or np.isnan(v).any()
            or np.isneginf(v).any() or not np.isfinite(w).all() or np.any(w < 0)
            or not w.sum() > 0 or not np.isfinite(w.sum())):
        raise ValueError("need aligned samples without NaN/-inf and finite nonnegative weights with positive mass")
    positive = w > 0
    return v[positive], w[positive] / w[positive].sum()


@dataclass(frozen=True)
class CellData:
    cell_id: str
    cycles: np.ndarray
    capacity: np.ndarray
    eol_cycle: Optional[float] = None   # None = no EOL observed within this record

    def __post_init__(self):
        if not isinstance(self.cell_id, str) or not self.cell_id.strip():
            raise ValueError("cell_id must be a nonempty string")
        c, y = _history(self.cycles, self.capacity)
        if self.eol_cycle is not None and (not np.isfinite(self.eol_cycle)
                or not c[0] <= self.eol_cycle <= c[-1]):
            raise ValueError("declared EOL must be inside the observed complete-cell record")
        # Own the arrays so mutating a caller's source cannot change a frozen record.
        c, y = c.copy(), y.copy()
        c.setflags(write=False)
        y.setflags(write=False)
        object.__setattr__(self, "cycles", c)
        object.__setattr__(self, "capacity", y)


# --------------------------------------------------------------------------- #
# Fade model
# --------------------------------------------------------------------------- #
def fade_curve(k, q0, s, g):
    k = np.asarray(k, dtype=float)
    g = np.asarray(g, dtype=float)
    small = np.abs(g) < 1e-7
    gg = np.where(small, 1.0, g)
    grow = np.where(small, k, np.expm1(np.clip(gg * k, -50, 50)) / gg)
    return q0 - s * grow


def crossing_cycle(q0, s, g, threshold: float) -> np.ndarray:
    """Cycle at which Q(k) first reaches `threshold` (inf if it never does)."""
    _positive(threshold, "threshold")
    q0, s, g = (np.atleast_1d(np.asarray(a, dtype=float)) for a in np.broadcast_arrays(q0, s, g))
    if not all(np.isfinite(a).all() for a in (q0, s, g)) or np.any(q0 <= 0) or np.any(s < 0):
        raise ValueError("fade parameters must be finite with q0>0 and s>=0")
    drop = q0 - threshold
    out = np.full(q0.shape, np.inf)
    valid = (s > 0) & (drop > 0)
    lin = valid & (np.abs(g) < 1e-7)
    out[lin] = drop[lin] / s[lin]
    nl = valid & ~lin
    arg = 1.0 + g[nl] * drop[nl] / s[nl]
    res = np.full(arg.shape, np.inf)
    ok = arg > 0
    res[ok] = np.log(arg[ok]) / g[nl][ok]
    out[nl] = res
    out[drop <= 0] = 0.0
    return out


def fit_fade(cycles, capacity) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares fit of (q0, s, g); returns (theta, covariance).

    Falls back to a linear fit (g = 0) when the nonlinear fit is ill-conditioned,
    which is common for short histories.
    """
    cycles, y = _history(cycles, capacity)
    k = cycles - cycles[0]
    slope0 = max((y[0] - y[-1]) / max(k[-1], 1.0), 1e-6)
    p0 = [y[: min(3, len(y))].mean(), slope0, 0.005]
    try:
        theta, cov = curve_fit(
            lambda kk, q0, s, g: fade_curve(kk, q0, s, g), k, y, p0=p0,
            bounds=([0.5 * y.max(), 1e-7, _G_MIN], [1.5 * y.max(), 1.0, _G_MAX]),
            maxfev=5000,
        )
        if not np.all(np.isfinite(cov)):
            raise RuntimeError("non-finite covariance")
        return theta, cov
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
        A = np.vstack([np.ones_like(k), -k]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        var = resid.var(ddof=2) if len(y) > 2 else 1e-4
        cov0 = var * np.linalg.pinv(A.T @ A)
        cov = np.zeros((3, 3))
        cov[:2, :2] = cov0
        cov[1, 1] = max(cov[1, 1], 1e-12)
        cov[2, 2] = 1e-6
        return np.array([coef[0], max(coef[1], 1e-7), 0.0]), cov


def weighted_quantiles(values, weights, qs: Sequence[float]) -> np.ndarray:
    """Weighted quantiles; +inf mass is handled (quantile is inf if it falls in it)."""
    v, w = _distribution(values, weights)
    qs = np.asarray(qs, float)
    if qs.ndim != 1 or not np.isfinite(qs).all() or np.any((qs < 0) | (qs > 1)):
        raise ValueError("quantile probabilities must be finite and in [0, 1]")
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cdf = np.cumsum(w) / w.sum()
    idx = np.minimum(np.searchsorted(cdf, qs, side="left"), len(v) - 1)
    return v[idx]


# --------------------------------------------------------------------------- #
# Particle filter
# --------------------------------------------------------------------------- #
class FadeParticleFilter:
    """Liu-West particle filter over (q0, log s, g)."""

    def __init__(
        self,
        eol_capacity: float,
        n_particles: int = 3000,
        prior: Optional[Dict[str, object]] = None,
        prior_inflation: float = 1.5,
        sigma_obs: Optional[float] = None,
        liu_west: float = 0.98,
        seed: int = 0,
    ):
        _positive(eol_capacity, "eol_capacity")
        _integer(n_particles, "n_particles", 2)
        _positive(prior_inflation, "prior_inflation")
        if sigma_obs is not None:
            _positive(sigma_obs, "sigma_obs")
        if not np.isfinite(liu_west) or not 0 < liu_west <= 1:
            raise ValueError("liu_west must be in (0, 1]")
        self.eol = eol_capacity
        self.n = n_particles
        self.prior = prior
        self.prior_inflation = prior_inflation
        self.sigma_obs = sigma_obs
        self.a = liu_west
        self.seed = seed
        self._fitted = False
        self.interval_calibrated_ = False

    @staticmethod
    def prior_from_cells(cells: Sequence[CellData], min_points: int = 15) -> Dict[str, object]:
        """Population prior in (q0, log s, g) space from training cells."""
        _integer(min_points, "min_points", 3)
        cells = list(cells)
        if len({c.cell_id for c in cells}) != len(cells):
            raise ValueError("training cell identities must be unique")
        th, identifiers = [], []
        for c in cells:
            if len(c.cycles) >= min_points:
                t, _ = fit_fade(c.cycles, c.capacity)
                th.append([t[0], np.log(t[1]), t[2]])
                identifiers.append(c.cell_id)
        if len(th) < 2:
            raise ValueError("need at least 2 training cells for a population prior")
        th = np.array(th)
        sd = th.std(0, ddof=1)
        floor = np.array([0.02, 0.15, 0.001])
        return {"mean": th.mean(0), "std": np.maximum(sd, floor),
                "training_cell_ids": tuple(identifiers)}

    # ------------------------------------------------------------------ #
    def fit(self, cycles, capacity, *, cell_id: Optional[str] = None) -> "FadeParticleFilter":
        self._fitted = False
        rng = np.random.default_rng(self.seed)
        c, y = _history(cycles, capacity)
        k0 = c[0]
        k = c - k0
        if self.sigma_obs is not None:
            sig = self.sigma_obs
        else:
            theta_hat, _ = fit_fade(c, y)
            resid = y - fade_curve(k, *theta_hat)
            sig = max(float(np.std(resid)), 0.002 * float(np.abs(y).mean()))

        if self.prior is not None:
            mean = np.asarray(self.prior["mean"], float)
            std = np.asarray(self.prior["std"], float) * self.prior_inflation
            if (mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all()
                    or not np.isfinite(std).all() or np.any(std <= 0)):
                raise ValueError("prior mean/std must be finite three-vectors with positive scales")
            training_ids = self.prior.get("training_cell_ids")
            if training_ids is not None:
                if not isinstance(cell_id, str) or not cell_id.strip():
                    raise ValueError("identified population priors require the held-out cell_id")
                if cell_id in training_ids:
                    raise ValueError("prediction cell must be disjoint from population-prior training cells")
        else:
            # A full-prefix fit is not a prior to which that same prefix can be
            # applied again. Use the first observation once and a broad fixed
            # fade prior; the rest of the prefix supplies the likelihood.
            mean = np.array([y[0], np.log(max(0.001 * y[0], 1e-7)), 0.003])
            std = np.array([max(0.1 * y[0], 0.05), 1.0, 0.01])

        def physical(particles):
            particles[:, 0] = np.maximum(particles[:, 0], 1e-7)
            particles[:, 1] = np.clip(particles[:, 1], np.log(1e-9), np.log(1.0))
            particles[:, 2] = np.clip(particles[:, 2], _G_MIN, _G_MAX)

        P = mean + std * rng.normal(size=(self.n, 3))
        physical(P)
        logw = np.zeros(self.n)
        h2 = 1.0 - self.a**2

        start = 0 if self.prior is not None else 1
        for ki, yi in zip(k[start:], y[start:]):
            pred = fade_curve(ki, P[:, 0], np.exp(P[:, 1]), P[:, 2])
            logw += -0.5 * ((yi - pred) / sig) ** 2
            if not np.isfinite(logw).any():
                raise ValueError("particle likelihood is numerically degenerate; inspect noise/prior settings")
            logw = np.where(np.isfinite(logw), logw, -np.inf)
            logw -= logw.max()
            w = np.exp(logw)
            w /= w.sum()
            if 1.0 / np.sum(w**2) < self.n / 2:
                mu = w @ P
                var = w @ (P - mu) ** 2
                # systematic resampling
                pos = (rng.random() + np.arange(self.n)) / self.n
                idx = np.minimum(np.searchsorted(np.cumsum(w), pos), self.n - 1)
                P = self.a * P[idx] + (1 - self.a) * mu \
                    + np.sqrt(h2 * var) * rng.normal(size=(self.n, 3))
                physical(P)
                logw = np.zeros(self.n)

        w = np.exp(logw - logw.max())
        self.weights_ = w / w.sum()
        self.particles_ = P
        self.k_now_ = k[-1]
        self.k0_ = k0
        self.sigma_obs_ = sig
        self.minimum_observed_capacity_ = float(y.min())
        self._fitted = True
        return self

    def rul_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("call fit() before predicting RUL")
        if self.minimum_observed_capacity_ <= self.eol:
            return np.zeros(self.n), self.weights_.copy()
        q0, s, g = self.particles_[:, 0], np.exp(self.particles_[:, 1]), self.particles_[:, 2]
        k_cross = crossing_cycle(q0, s, g, self.eol)
        return np.maximum(k_cross - self.k_now_, 0.0), self.weights_

    def rul_quantiles(self, qs=(0.05, 0.5, 0.95)) -> np.ndarray:
        r, w = self.rul_samples()
        return weighted_quantiles(r, w, qs)


# --------------------------------------------------------------------------- #
# Gaussian process
# --------------------------------------------------------------------------- #
class GPFadeModel:
    """Parametric fade plus latent-GP residual, with a finite forecast horizon.

    An infinite sample/quantile means no crossing found by censoring_horizon_,
    not a known infinite lifetime. Do not use that tail to assert survival beyond
    the horizon. The returned uncertainty is approximate and uncalibrated.
    """

    def __init__(self, eol_capacity: float, horizon: int = 300, n_samples: int = 200,
                 cov_inflation: float = 2.0, seed: int = 0):
        _positive(eol_capacity, "eol_capacity")
        _integer(horizon, "horizon")
        _integer(n_samples, "n_samples")
        _positive(cov_inflation, "cov_inflation")
        self.eol, self.horizon, self.n_samples = eol_capacity, horizon, n_samples
        self.cov_inflation, self.seed = cov_inflation, seed
        self._fitted = False
        self.interval_calibrated_ = False
        self.censoring_horizon_ = float(horizon)

    def fit(self, cycles, capacity) -> "GPFadeModel":
        self._fitted = False
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel

        c, y = _history(cycles, capacity)
        self.k0_ = c[0]
        k = c - self.k0_
        self.theta_, self.cov_ = fit_fade(c, y)
        resid = y - fade_curve(k, *self.theta_)
        scale = max(float(resid.std()), 1e-4)
        kernel = C(scale**2, (1e-8, 1.0)) * RBF(10.0, (3.0, 100.0)) + WhiteKernel(scale**2, (1e-8, 1.0))
        self.gp_ = GaussianProcessRegressor(kernel, normalize_y=False,
                                            n_restarts_optimizer=1, random_state=self.seed)
        self.gp_.fit(k[:, None], resid)
        self.k_now_ = k[-1]
        self.minimum_observed_capacity_ = float(y.min())
        self._fitted = True
        return self

    def rul_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("call fit() before predicting RUL")
        rng = np.random.default_rng(self.seed)
        n, H = self.n_samples, self.horizon
        self.censoring_horizon_ = float(H)
        if self.minimum_observed_capacity_ <= self.eol:
            return np.zeros(n), np.full(n, 1.0 / n)
        grid = self.k_now_ + np.arange(H + 1, dtype=float)
        size = len(grid)
        # white noise is observation noise, not part of the latent degradation path
        from scipy.linalg import solve_triangular

        latent = self.gp_.kernel_.k1
        K = latent(grid[:, None]) + 1e-10 * np.eye(size)
        Ks = latent(grid[:, None], self.gp_.X_train_)
        A = self.gp_.alpha_
        mean_res = Ks @ A
        v = solve_triangular(self.gp_.L_, Ks.T, lower=True)
        cov_res = K - v.T @ v
        cov_res = 0.5 * (cov_res + cov_res.T) + 1e-10 * np.eye(size)
        res_samples = rng.multivariate_normal(mean_res, cov_res, size=n, method="svd")

        thetas = rng.multivariate_normal(self.theta_, self.cov_ * self.cov_inflation, size=n)
        thetas[:, 0] = np.maximum(thetas[:, 0], self.eol + 1e-6)
        thetas[:, 1] = np.maximum(thetas[:, 1], 1e-7)
        thetas[:, 2] = np.clip(thetas[:, 2], _G_MIN, _G_MAX)

        rul = np.full(n, np.inf)
        for i in range(n):
            traj = fade_curve(grid, *thetas[i]) + res_samples[i]
            hit = np.where(traj <= self.eol)[0]
            if hit.size:
                rul[i] = float(hit[0])
        return rul, np.full(n, 1.0 / n)

    def rul_quantiles(self, qs=(0.05, 0.5, 0.95)) -> np.ndarray:
        r, w = self.rul_samples()
        return weighted_quantiles(r, w, qs)


# --------------------------------------------------------------------------- #
# Prognostic metrics
# --------------------------------------------------------------------------- #
def alpha_lambda_hits(rul_true, rul_pred, alpha: float = 0.2) -> np.ndarray:
    """True where the prediction is within +/- alpha * true RUL."""
    t, p = np.asarray(rul_true, float), np.asarray(rul_pred, float)
    if (t.ndim != 1 or not len(t) or t.shape != p.shape or not np.isfinite(t).all()
            or np.any(t < 0) or np.isnan(p).any() or np.any(p < 0)
            or not np.isfinite(alpha) or not 0 <= alpha < 1):
        raise ValueError("need aligned nonnegative RUL vectors and alpha in [0, 1)")
    return np.abs(p - t) <= alpha * np.maximum(t, 1e-9)


def prognostic_horizon(rul_true, rul_pred, alpha: float = 0.2) -> float:
    """Remaining life (cycles) at the earliest point after which every prediction
    stays inside the alpha band; 0 if the final prediction is outside it."""
    hits = alpha_lambda_hits(rul_true, rul_pred, alpha)
    if np.any(np.diff(np.asarray(rul_true, float)) > 0):
        raise ValueError("prognostic horizon requires chronological forecasts with decreasing true RUL")
    if not hits[-1]:
        return 0.0
    first = len(hits) - 1
    while first > 0 and hits[first - 1]:
        first -= 1
    return float(np.asarray(rul_true, float)[first])


def survival_consistency(rul_samples, weights, remaining_observed: float,
                         *, censoring_horizon: Optional[float] = None) -> float:
    """P(RUL > remaining observed life) for a cell that was still alive at the
    end of its record.  Close to 1 is consistent; low values mean the model
    predicted a failure that demonstrably did not occur."""
    r, w = _distribution(rul_samples, weights)
    if np.any(r < 0) or not np.isfinite(remaining_observed) or remaining_observed < 0:
        raise ValueError("remaining observed life and RUL samples must be nonnegative")
    if censoring_horizon is not None:
        _positive(censoring_horizon, "censoring_horizon")
        if remaining_observed > censoring_horizon:
            raise ValueError("survival beyond the prediction horizon is unscorable")
    return float(w[r > remaining_observed].sum())


# --------------------------------------------------------------------------- #
# Leave-one-cell-out evaluation
# --------------------------------------------------------------------------- #
def leave_one_cell_out(
    cells: Sequence[CellData],
    eol_capacity: float,
    method: str = "pf",
    start_fractions: Sequence[float] = (0.4, 0.5, 0.6, 0.7, 0.8),
    alpha: float = 0.2,
    seed: int = 0,
    **kwargs,
) -> Dict[str, object]:
    """Hold out each cell; predict RUL from its history up to several cutoffs.

    The population prior (particle filter) is built from the OTHER cells only.
    """
    cells = list(cells)
    if not cells or len({c.cell_id for c in cells}) != len(cells):
        raise ValueError("leave-one-cell-out requires nonempty unique cell identities")
    _positive(eol_capacity, "eol_capacity")
    if method not in {"pf", "gp"}:
        raise ValueError("method must be 'pf' or 'gp'")
    if not np.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1)")
    fractions = np.asarray(start_fractions, float)
    if (fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all()
            or np.any((fractions <= 0) | (fractions >= 1))
            or len(np.unique(fractions)) != len(fractions)):
        raise ValueError("start_fractions must be distinct finite fractions in (0, 1)")
    for cell in cells:
        if cell.eol_cycle is None and np.any(cell.capacity <= eol_capacity):
            raise ValueError("a censored record must not contain an observed EOL crossing")
    rows, censored_rows = [], []
    for held in cells:
        others = [c for c in cells if c.cell_id != held.cell_id]
        prior = FadeParticleFilter.prior_from_cells(others) if method == "pf" else None
        n_obs = len(held.cycles)
        life = held.eol_cycle
        for frac in fractions:
            end = int(frac * (n_obs if life is None else np.searchsorted(held.cycles, life) + 1))
            if life is not None:
                end = min(end, int(np.searchsorted(held.cycles, life, side="left")))
            if end < 10:
                continue
            if end >= n_obs:
                continue
            cyc, cap = held.cycles[:end], held.capacity[:end]
            if method == "pf":
                model = FadeParticleFilter(eol_capacity, prior=prior, seed=seed, **kwargs).fit(
                    cyc, cap, cell_id=held.cell_id)
            elif method == "gp":
                model = GPFadeModel(eol_capacity, seed=seed, **kwargs).fit(cyc, cap)
            else:
                raise ValueError("method must be 'pf' or 'gp'")
            r, w = model.rul_samples()
            lo, med, hi = weighted_quantiles(r, w, (0.05, 0.5, 0.95))
            now = float(cyc[-1])
            if life is None:
                remaining = float(held.cycles[-1] - now)
                horizon = getattr(model, "censoring_horizon_", None)
                scorable = horizon is None or remaining <= horizon
                censored_rows.append({"cell": held.cell_id, "frac": float(frac),
                    "remaining_observed": remaining, "prediction_horizon": horizon,
                    "status": "ok" if scorable else "unscorable_beyond_prediction_horizon",
                    "consistency": survival_consistency(r, w, remaining, censoring_horizon=horizon)
                                   if scorable else None})
            else:
                true = float(life - now)
                horizon = getattr(model, "censoring_horizon_", None)
                interval_scorable = horizon is None or true <= horizon
                rows.append({"cell": held.cell_id, "frac": frac, "true": true, "median": float(med),
                             "lo": float(lo), "hi": float(hi),
                             "tail_censored_at": horizon,
                             "interval_scorable": interval_scorable,
                             "interval_contains_truth": bool(lo <= true <= hi) if interval_scorable else None,
                             "point_available": bool(np.isfinite(med))})

    out: Dict[str, object] = {"rows": rows, "censored_rows": censored_rows,
        "method": method, "interval_calibrated": False, "evaluation_unit": "cell",
        "cutoff_protocol": "retrospective_fractions_of_recorded_life", "n_cells": len(cells)}
    if rows:
        t = np.array([r["true"] for r in rows])
        m = np.array([r["median"] for r in rows])
        hi = np.array([r["hi"] for r in rows])
        finite = np.isfinite(m)
        interval_hits = [row["interval_contains_truth"] for row in rows if row["interval_scorable"]]
        out.update(
            n=len(rows),
            n_point_forecasts=int(finite.sum()), point_forecast_coverage=float(finite.mean()),
            median_abs_error=float(np.median(np.abs(m[finite] - t[finite]))) if finite.any() else None,
            median_rel_error=float(np.median(np.abs(m[finite] - t[finite]) / np.maximum(t[finite], 1.0)))
                             if finite.any() else None,
            alpha_lambda_accuracy=float(alpha_lambda_hits(t, m, alpha).mean()),
            interval90_coverage=float(np.mean(interval_hits)) if interval_hits else None,
            n_interval_scorable=len(interval_hits), n_interval_unscorable=len(rows) - len(interval_hits),
            unbounded_interval_fraction=float((~np.isfinite(hi)).mean()),
        )
        per_cell = {}
        for cid in sorted({r["cell"] for r in rows}):
            sel = [r for r in rows if r["cell"] == cid]
            errors = [abs(r["median"] - r["true"]) for r in sel if r["point_available"]]
            per_cell[cid] = float(np.median(errors)) if errors else None
        out["per_cell_median_abs_error"] = per_cell
        finite_cell_errors = [error for error in per_cell.values() if error is not None]
        out["n_cells_with_point_metrics"] = len(finite_cell_errors)
        out["macro_cell_median_abs_error"] = float(np.mean(finite_cell_errors)) if finite_cell_errors else None
    if censored_rows:
        scored = [r["consistency"] for r in censored_rows if r["consistency"] is not None]
        out["n_censored_unscorable"] = len(censored_rows) - len(scored)
        out["censored_mean_consistency"] = float(np.mean(scored)) if scored else None
    return out


# --------------------------------------------------------------------------- #
# Discharge-curve features
# --------------------------------------------------------------------------- #
def _discharge_arrays(t, v, i=None, minimum=2):
    t, v = np.asarray(t, float), np.asarray(v, float)
    if (t.ndim != 1 or t.shape != v.shape or len(t) < minimum
            or not np.isfinite(t).all() or not np.isfinite(v).all()
            or np.any(t < 0) or np.any(np.diff(t) <= 0) or np.any(v <= 0)):
        raise ValueError("need aligned finite positive voltage and increasing nonnegative time")
    if i is None:
        return t, v
    i = np.asarray(i, float)
    if i.shape != t.shape or not np.isfinite(i).all() or not np.any(i):
        raise ValueError("discharge current must be finite, aligned and nonzero")
    if np.any(i > 0) and np.any(i < 0):
        raise ValueError("separate charge and discharge segments before computing incremental capacity")
    return t, v, i


def time_to_voltage(t, v, v_threshold: float) -> float:
    """First time the (discharge) voltage falls to v_threshold; NaN if it never does."""
    t, v = _discharge_arrays(t, v)
    _positive(v_threshold, "v_threshold")
    below = np.where(v <= v_threshold)[0]
    if below.size == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float(t[0])
    frac = (v[i - 1] - v_threshold) / (v[i - 1] - v[i])
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def incremental_capacity(t, v, i, n_grid: int = 200, window: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    """dQ/dV (Ah/V) of one discharge.  t [s], v [V], i discharge current [A].

    Capacity is integrated from the current, voltage is mapped onto a uniform
    grid and both are Savitzky-Golay smoothed before differentiating.
    """
    t, v, i = _discharge_arrays(t, v, i, minimum=5)
    _integer(n_grid, "n_grid", 5)
    _integer(window, "window", 5)
    if window % 2 == 0:
        raise ValueError("smoothing window must be odd")
    if np.any(np.diff(v) > 0) or v[-1] >= v[0]:
        raise ValueError("incremental capacity requires a cleaned monotone discharge-voltage segment")
    q = np.concatenate([[0.0], np.cumsum(0.5 * (np.abs(i[1:]) + np.abs(i[:-1])) * np.diff(t))]) / 3600.0
    order = np.argsort(v, kind="stable")       # monotone voltage, ascending for interpolation
    vs, qs = v[order], q[order]
    vs, inverse, count = np.unique(vs, return_inverse=True, return_counts=True)
    qs = np.bincount(inverse, weights=qs) / count
    if len(vs) < 4:
        raise ValueError("incremental capacity requires at least four distinct voltages")
    grid = np.linspace(vs[0], vs[-1], n_grid)
    q_grid = np.interp(grid, vs, qs)
    win = min(window, n_grid if n_grid % 2 else n_grid - 1)
    q_s = savgol_filter(q_grid, win, 3)
    ic = -np.gradient(q_s, grid)                # capacity released per volt dropped
    return grid, savgol_filter(ic, win, 3)


def ic_peak(v_grid, ic) -> Tuple[float, float]:
    """(peak height, peak voltage) of an incremental-capacity curve."""
    v_grid, ic = np.asarray(v_grid, float), np.asarray(ic, float)
    if (v_grid.ndim != 1 or len(v_grid) < 3 or v_grid.shape != ic.shape
            or not np.isfinite(v_grid).all() or not np.isfinite(ic).all()
            or np.any(np.diff(v_grid) <= 0) or not np.any(ic > 0)):
        raise ValueError("need aligned finite increasing voltage and a positive IC peak")
    j = int(np.argmax(ic))
    return float(ic[j]), float(v_grid[j])
