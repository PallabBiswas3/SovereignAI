from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("expected [samples, channels]")
    if not np.all(np.isfinite(arr)):
        raise ValueError("input contains NaN or Inf")
    return arr


def apply_alarm_persistence(alarm_mask, min_consecutive: int = 1) -> np.ndarray:
    """Keep alarms only after a run persists for min_consecutive samples.

    This is a simple false-alarm-control baseline. It suppresses isolated spikes but
    preserves sustained process deviations.
    """
    alarm = np.asarray(alarm_mask, dtype=bool).ravel()
    k = max(1, int(min_consecutive))
    if k <= 1:
        return alarm.copy()
    out = np.zeros_like(alarm)
    run = 0
    for i, active in enumerate(alarm):
        if active:
            run += 1
            if run >= k:
                out[i] = True
        else:
            run = 0
    return out


def make_lagged_matrix(signal_matrix, lags: int = 1) -> np.ndarray:
    """Construct a dynamic/lagged matrix [x_t, x_{t-1}, ..., x_{t-lags}]."""
    x = _as_2d(signal_matrix)
    p = max(0, int(lags))
    if p == 0:
        return x.copy()
    if x.shape[0] <= p:
        raise ValueError("not enough samples for requested lags")
    rows = []
    for lag in range(p + 1):
        rows.append(x[p - lag : x.shape[0] - lag])
    return np.hstack(rows)


def make_cva_past_future(
    signal_matrix,
    past_lags: int = 2,
    future_lags: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build causal past blocks and training-only future blocks for CVA.

    ``indices`` identifies the sample at the end of each past block.  Future
    blocks are used only to learn the canonical projection; online scoring uses
    the past block and therefore does not look ahead.
    """
    x = _as_2d(signal_matrix)
    p = max(1, int(past_lags))
    f = max(1, int(future_lags))
    if len(x) < p + f:
        raise ValueError("not enough samples for requested CVA past/future windows")
    indices = np.arange(p - 1, len(x) - f, dtype=int)
    past = np.stack(
        [np.concatenate([x[t - lag] for lag in range(p)]) for t in indices]
    )
    future = np.stack(
        [np.concatenate([x[t + lead] for lead in range(1, f + 1)]) for t in indices]
    )
    return past, future, indices


def _cva_past_blocks(signal_matrix, past_lags: int) -> tuple[np.ndarray, np.ndarray]:
    x = _as_2d(signal_matrix)
    p = max(1, int(past_lags))
    if len(x) < p:
        raise ValueError("not enough samples for requested CVA past window")
    indices = np.arange(p - 1, len(x), dtype=int)
    past = np.stack(
        [np.concatenate([x[t - lag] for lag in range(p)]) for t in indices]
    )
    return past, indices


def _inverse_sqrt_psd(matrix: np.ndarray, regularization: float) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    floor = max(float(regularization), float(np.max(values, initial=0.0)) * regularization)
    return (vectors * (1.0 / np.sqrt(np.maximum(values, floor)))) @ vectors.T


def fit_cva(
    normal_reference,
    *,
    past_lags: int = 2,
    future_lags: int = 2,
    variance_target: float = 0.95,
    regularization: float = 1e-6,
) -> dict[str, Any]:
    """Fit a linear canonical-variate state model from healthy operation only."""
    ref = _as_2d(normal_reference)
    past, future, _ = make_cva_past_future(ref, past_lags, future_lags)
    past_mean = past.mean(axis=0)
    future_mean = future.mean(axis=0)
    pc = past - past_mean
    fc = future - future_mean
    denom = max(len(pc) - 1, 1)
    cov_pp = pc.T @ pc / denom
    cov_ff = fc.T @ fc / denom
    cov_pf = pc.T @ fc / denom
    whiten_p = _inverse_sqrt_psd(cov_pp, regularization)
    whiten_f = _inverse_sqrt_psd(cov_ff, regularization)
    left, canonical_correlations, _ = np.linalg.svd(
        whiten_p @ cov_pf @ whiten_f, full_matrices=False
    )
    energy = np.square(canonical_correlations)
    if float(np.sum(energy)) <= 1e-12:
        n_components = 1
    else:
        cumulative = np.cumsum(energy) / np.sum(energy)
        n_components = int(np.searchsorted(cumulative, variance_target) + 1)
    n_components = min(max(1, n_components), left.shape[1])
    state_basis = left[:, :n_components]
    return {
        "past_lags": max(1, int(past_lags)),
        "future_lags": max(1, int(future_lags)),
        "channel_count": int(ref.shape[1]),
        "past_mean": past_mean,
        "past_whitener": whiten_p,
        "state_basis": state_basis,
        "canonical_correlations": canonical_correlations,
        "n_components": n_components,
        "variance_target": float(variance_target),
        "regularization": float(regularization),
    }


def transform_cva(signal_matrix, cva_state: dict[str, Any]) -> dict[str, Any]:
    """Project a sequence into causal CVA state and residual features."""
    x = _as_2d(signal_matrix)
    if x.shape[1] != int(cva_state["channel_count"]):
        raise ValueError("CVA input channel count differs from fitted state")
    p = int(cva_state["past_lags"])
    blocks, indices = _cva_past_blocks(x, p)
    centered = blocks - np.asarray(cva_state["past_mean"])
    whitener = np.asarray(cva_state["past_whitener"])
    basis = np.asarray(cva_state["state_basis"])
    white = centered @ whitener
    scores = white @ basis
    residual_white = white - scores @ basis.T
    residual_blocks = residual_white @ np.linalg.pinv(whitener)
    channel_contributions = residual_blocks.reshape(len(blocks), p, x.shape[1])
    channel_contributions = np.sum(np.square(channel_contributions), axis=1)
    return {
        "scores": scores,
        "residual_scores": residual_white,
        "channel_contributions": channel_contributions,
        "sample_indices": indices,
    }


def cva_monitoring(
    standardized_signal,
    normal_reference,
    *,
    past_lags: int = 2,
    future_lags: int = 2,
    variance_target: float = 0.95,
    alpha: float = 0.99,
    min_consecutive: int = 1,
    regularization: float = 1e-6,
) -> dict[str, Any]:
    """CVA state/residual monitoring with healthy-reference control limits."""
    current = _as_2d(standardized_signal)
    reference = _as_2d(normal_reference)
    state = fit_cva(
        reference,
        past_lags=past_lags,
        future_lags=future_lags,
        variance_target=variance_target,
        regularization=regularization,
    )
    ref_features = transform_cva(reference, state)
    cur_features = transform_cva(current, state)
    ref_scores = ref_features["scores"]
    score_cov = np.atleast_2d(np.cov(ref_scores, rowvar=False))
    score_cov += regularization * np.eye(score_cov.shape[0])
    inv_score_cov = np.linalg.pinv(score_cov)
    ref_t2 = np.einsum("ij,jk,ik->i", ref_scores, inv_score_cov, ref_scores)
    t2 = np.einsum("ij,jk,ik->i", cur_features["scores"], inv_score_cov, cur_features["scores"])
    ref_q = np.sum(np.square(ref_features["residual_scores"]), axis=1)
    q = np.sum(np.square(cur_features["residual_scores"]), axis=1)
    limits = {"t2": float(np.quantile(ref_t2, alpha)), "spe": float(np.quantile(ref_q, alpha))}
    raw_core = (t2 > limits["t2"]) | (q > limits["spe"])
    alarm_core = apply_alarm_persistence(raw_core, min_consecutive)
    combined = np.maximum(t2 / (limits["t2"] + 1e-12), q / (limits["spe"] + 1e-12))
    pad_n = int(cur_features["sample_indices"][0])
    pad_bool = np.zeros(pad_n, dtype=bool)
    pad_float = np.full(pad_n, np.nan)
    contributions = np.zeros((len(current), current.shape[1]), dtype=float)
    contributions[pad_n:] = cur_features["channel_contributions"]
    return {
        "scores": cur_features["scores"],
        "t2": np.r_[pad_float, t2],
        "spe": np.r_[pad_float, q],
        "combined_score": np.r_[pad_float, combined],
        "control_limits": limits,
        "raw_alarm_mask": np.r_[pad_bool, raw_core],
        "alarm_mask": np.r_[pad_bool, alarm_core],
        "variable_contributions": contributions,
        "cva_state": state,
        "sample_indices": cur_features["sample_indices"],
        "calibration": {
            "method": "cva",
            "alpha": float(alpha),
            "min_consecutive": int(min_consecutive),
            "variance_target": float(variance_target),
            "lags": int(past_lags),
            "future_lags": int(future_lags),
        },
    }


def arbitrate_dpca_cva(
    dpca_result: dict[str, Any],
    cva_result: dict[str, Any],
    *,
    minimum_alarm_fraction: float = 0.05,
    strong_dpca_alarm_fraction: float = 0.20,
) -> dict[str, Any]:
    """Combine fast DPCA warning evidence with conservative CVA confirmation.

    CVA confirmation is sufficient for a fault decision. DPCA-only evidence is
    an early warning unless it is both strong and persistent over a substantial
    fraction of the sequence. The rule is fixed independently of fault labels.
    """
    dpca_mask = np.asarray(dpca_result["alarm_mask"], dtype=bool)
    cva_mask = np.asarray(cva_result["alarm_mask"], dtype=bool)
    if dpca_mask.shape != cva_mask.shape:
        raise ValueError("DPCA and CVA alarm masks must align")
    dpca_fraction = float(np.mean(dpca_mask))
    cva_fraction = float(np.mean(cva_mask))
    threshold = float(minimum_alarm_fraction)
    dpca_warning = dpca_fraction >= threshold
    cva_confirmed = cva_fraction >= threshold
    strong_dpca = dpca_fraction >= max(float(strong_dpca_alarm_fraction), threshold)
    confirmed = cva_confirmed or strong_dpca
    if cva_confirmed and dpca_warning:
        status, reason = "confirmed_fault", "dpca_cva_agreement"
    elif cva_confirmed:
        status, reason = "confirmed_fault", "cva_confirmation"
    elif strong_dpca:
        status, reason = "confirmed_fault", "strong_persistent_dpca_override"
    elif dpca_warning:
        status, reason = "early_warning", "dpca_warning_awaiting_cva_confirmation"
    else:
        status, reason = "normal", "no_persistent_dynamic_alarm"
    return {
        "status": status,
        "reason": reason,
        "fault_detected": bool(confirmed),
        "early_warning": bool(status == "early_warning"),
        "dpca_alarm_fraction": dpca_fraction,
        "cva_alarm_fraction": cva_fraction,
        "minimum_alarm_fraction": threshold,
        "strong_dpca_alarm_fraction": float(strong_dpca_alarm_fraction),
        "alarm_mask": cva_mask if cva_confirmed else dpca_mask,
    }


def _choose_n_components(reference: np.ndarray, variance_target: float) -> int:
    full = PCA().fit(reference)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_target) + 1)
    return min(max(1, n_components), min(reference.shape))


def _monitoring_core(
    current_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    variance_target: float = 0.95,
    alpha: float = 0.99,
    min_consecutive: int = 1,
) -> dict[str, Any]:
    cur = _as_2d(current_features)
    ref = _as_2d(reference_features)
    if cur.shape[1] != ref.shape[1]:
        raise ValueError("current and reference dimensions differ")

    n_components = _choose_n_components(ref, variance_target)
    pca = PCA(n_components=n_components).fit(ref)

    ref_scores = pca.transform(ref)
    cur_scores = pca.transform(cur)
    ref_recon = pca.inverse_transform(ref_scores)
    cur_recon = pca.inverse_transform(cur_scores)

    ref_spe = np.sum((ref - ref_recon) ** 2, axis=1)
    spe = np.sum((cur - cur_recon) ** 2, axis=1)

    score_mu = ref_scores.mean(axis=0)
    score_cov = np.atleast_2d(np.cov(ref_scores, rowvar=False)) + 1e-9 * np.eye(n_components)
    inv_cov = np.linalg.pinv(score_cov)
    ref_t2 = np.einsum("ij,jk,ik->i", ref_scores - score_mu, inv_cov, ref_scores - score_mu)
    t2 = np.einsum("ij,jk,ik->i", cur_scores - score_mu, inv_cov, cur_scores - score_mu)

    limits = {
        "spe": float(np.quantile(ref_spe, alpha)),
        "t2": float(np.quantile(ref_t2, alpha)),
    }
    raw_alarm = (spe > limits["spe"]) | (t2 > limits["t2"])
    alarm = apply_alarm_persistence(raw_alarm, min_consecutive)
    score = np.maximum(t2 / (limits["t2"] + 1e-12), spe / (limits["spe"] + 1e-12))

    return {
        "scores": cur_scores,
        "t2": t2,
        "spe": spe,
        "combined_score": score,
        "control_limits": limits,
        "raw_alarm_mask": raw_alarm,
        "alarm_mask": alarm,
        "pca_state": {
            "components": pca.components_,
            "mean": pca.mean_,
            "explained_variance": pca.explained_variance_,
            "n_components": n_components,
        },
        "calibration": {
            "alpha": float(alpha),
            "min_consecutive": int(min_consecutive),
            "variance_target": float(variance_target),
        },
    }


def calibrated_pca_monitoring(
    standardized_signal,
    normal_reference,
    *,
    variance_target: float = 0.95,
    alpha: float = 0.99,
    min_consecutive: int = 1,
) -> dict[str, Any]:
    return _monitoring_core(
        standardized_signal,
        normal_reference,
        variance_target=variance_target,
        alpha=alpha,
        min_consecutive=min_consecutive,
    )


def dpca_monitoring(
    standardized_signal,
    normal_reference,
    *,
    lags: int = 2,
    variance_target: float = 0.95,
    alpha: float = 0.99,
    min_consecutive: int = 1,
) -> dict[str, Any]:
    """Dynamic PCA baseline using lagged variables.

    DPCA often reduces false alarms in autocorrelated process data because the PCA
    model sees short temporal context rather than isolated samples.
    """
    x = _as_2d(standardized_signal)
    ref = _as_2d(normal_reference)
    p = max(0, int(lags))
    cur_features = make_lagged_matrix(x, p)
    ref_features = make_lagged_matrix(ref, p)
    core = _monitoring_core(
        cur_features,
        ref_features,
        variance_target=variance_target,
        alpha=alpha,
        min_consecutive=min_consecutive,
    )

    pad = np.zeros(p, dtype=bool)
    pad_float = np.full(p, np.nan)
    core["raw_alarm_mask"] = np.r_[pad, core["raw_alarm_mask"]]
    core["alarm_mask"] = np.r_[pad, core["alarm_mask"]]
    core["t2"] = np.r_[pad_float, core["t2"]]
    core["spe"] = np.r_[pad_float, core["spe"]]
    core["combined_score"] = np.r_[pad_float, core["combined_score"]]
    core["dpca_lags"] = p
    core["calibration"]["method"] = "dpca"
    core["calibration"]["lags"] = p
    return core


@dataclass(frozen=True)
class MonitoringConfig:
    method: str
    alpha: float
    min_consecutive: int
    variance_target: float = 0.95
    lags: int = 0
    validation_false_alarm_rate: float | None = None


def run_monitoring_method(
    standardized_signal,
    standardized_reference,
    config: MonitoringConfig,
) -> dict[str, Any]:
    method = config.method.lower()
    if method == "pca":
        result = calibrated_pca_monitoring(
            standardized_signal,
            standardized_reference,
            variance_target=config.variance_target,
            alpha=config.alpha,
            min_consecutive=config.min_consecutive,
        )
    elif method == "dpca":
        result = dpca_monitoring(
            standardized_signal,
            standardized_reference,
            lags=config.lags,
            variance_target=config.variance_target,
            alpha=config.alpha,
            min_consecutive=config.min_consecutive,
        )
    elif method == "cva":
        result = cva_monitoring(
            standardized_signal,
            standardized_reference,
            past_lags=max(1, config.lags),
            future_lags=max(1, config.lags),
            variance_target=config.variance_target,
            alpha=config.alpha,
            min_consecutive=config.min_consecutive,
        )
    else:
        raise ValueError("method must be 'pca', 'dpca', or 'cva'")
    result["calibration"]["method"] = method
    return result


def calibrate_monitoring_config(
    standardized_reference,
    *,
    method: str = "pca",
    variance_target: float = 0.95,
    target_false_alarm_rate: float = 0.05,
    calibration_fraction: float = 0.35,
    alpha_grid: Iterable[float] = (0.99, 0.995, 0.9975, 0.999, 0.9995),
    persistence_grid: Iterable[int] = (1, 2, 3, 5),
    lags_grid: Iterable[int] = (0, 1, 2, 3),
) -> dict[str, Any]:
    """Tune monitoring thresholds on held-out healthy data.

    This targets false-alarm control only. It does not use fault labels, so it is a
    fair normal-operation calibration stage.
    """
    ref = _as_2d(standardized_reference)
    if ref.shape[0] < 40:
        raise ValueError("need at least 40 healthy samples for calibration")

    frac = min(max(float(calibration_fraction), 0.15), 0.5)
    split = max(20, int(round(ref.shape[0] * (1.0 - frac))))
    split = min(split, ref.shape[0] - 10)
    fit_ref = ref[:split]
    val_ref = ref[split:]

    method = method.lower()
    if method not in {"pca", "dpca", "cva"}:
        raise ValueError("method must be 'pca', 'dpca', or 'cva'")
    lags_candidates = [0] if method == "pca" else [max(1, int(v)) for v in lags_grid]
    trials = []
    for lags in lags_candidates:
        if method in {"dpca", "cva"} and min(len(fit_ref), len(val_ref)) <= 2 * lags + 5:
            continue
        for alpha in alpha_grid:
            for persistence in persistence_grid:
                cfg = MonitoringConfig(
                    method=method,
                    alpha=float(alpha),
                    min_consecutive=int(persistence),
                    variance_target=float(variance_target),
                    lags=int(lags),
                )
                try:
                    monitored = run_monitoring_method(val_ref, fit_ref, cfg)
                    far = float(np.nanmean(np.asarray(monitored["alarm_mask"], dtype=bool)))
                except Exception:
                    continue
                trials.append({"config": cfg, "validation_false_alarm_rate": far})

    if not trials:
        raise RuntimeError("no valid monitoring calibration trials")

    target = float(target_false_alarm_rate)

    def key(row):
        far = row["validation_false_alarm_rate"]
        over = far > target
        # Prefer settings close to the target from below. If every setting is over,
        # choose the lowest false-alarm rate.
        return (over, abs(far - target) if not over else far, row["config"].min_consecutive)

    best = sorted(trials, key=key)[0]
    cfg = best["config"]
    chosen = MonitoringConfig(
        method=cfg.method,
        alpha=cfg.alpha,
        min_consecutive=cfg.min_consecutive,
        variance_target=cfg.variance_target,
        lags=cfg.lags,
        validation_false_alarm_rate=best["validation_false_alarm_rate"],
    )
    return {
        "config": chosen,
        "trials": [
            {
                "method": row["config"].method,
                "alpha": row["config"].alpha,
                "min_consecutive": row["config"].min_consecutive,
                "lags": row["config"].lags,
                "validation_false_alarm_rate": row["validation_false_alarm_rate"],
            }
            for row in trials
        ],
    }
