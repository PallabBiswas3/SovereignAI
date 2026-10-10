from __future__ import annotations

import numpy as np


def _history(cycle_index, capacity_ah) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(cycle_index, dtype=float).ravel()
    y = np.asarray(capacity_ah, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("cycle_index and capacity_ah must have equal length")
    if not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(np.diff(x) <= 0):
        raise ValueError("cycle_index must be finite, nonnegative and strictly increasing")
    if np.any(np.isinf(y)) or np.any(y[np.isfinite(y)] <= 0):
        raise ValueError("capacity must be positive; missing readings must use NaN")
    finite = np.isfinite(y)
    x, y = x[finite], y[finite]
    return x, y


def smooth_capacity(capacity_ah, window: int = 5) -> np.ndarray:
    y = np.asarray(capacity_ah, dtype=float).ravel()
    if y.size < 3 or window <= 1:
        return y.copy()
    radius = max(1, int(window) // 2)
    return np.asarray([
        np.median(y[max(0, i-radius):min(len(y), i+radius+1)])
        for i in range(len(y))
    ], dtype=float)


def capacity_features(cycle_index, capacity_ah, *, nominal_capacity_ah=2.0, slope_window=20) -> dict:
    if not np.isfinite(nominal_capacity_ah) or nominal_capacity_ah <= 0:
        raise ValueError("nominal_capacity_ah must be positive and finite")
    raw_y = np.asarray(capacity_ah, dtype=float).ravel()
    x, y = _history(cycle_index, raw_y)
    smooth = smooth_capacity(y)
    if x.size < 2:
        return {
            "cycle_count": int(x.size),
            "current_capacity_ah": None if not x.size else float(y[-1]),
            "soh": None if not x.size else float(y[-1] / nominal_capacity_ah),
            "local_slope_ah_per_cycle": None,
            "global_slope_ah_per_cycle": None,
            "smoothed_capacity_ah": smooth.tolist(),
        }
    n = min(max(3, int(slope_window)), x.size)
    return {
        "cycle_count": int(x.size),
        "current_capacity_ah": float(y[-1]),
        "soh": float(y[-1] / nominal_capacity_ah),
        "local_slope_ah_per_cycle": float(np.polyfit(x[-n:], smooth[-n:], 1)[0]),
        "global_slope_ah_per_cycle": float(np.polyfit(x, smooth, 1)[0]),
        "smoothed_capacity_ah": smooth.tolist(),
    }


def _fit_crossing(x: np.ndarray, y: np.ndarray, threshold: float):
    slope, intercept = np.polyfit(x, y, 1)
    slope, intercept = float(slope), float(intercept)
    if slope >= -1e-5:
        return None
    eol = float((threshold - intercept) / slope)
    return slope, intercept, eol


def _crossing_parameter_uncertainty(
    x: np.ndarray,
    y: np.ndarray,
    slope: float,
    intercept: float,
    threshold: float,
    confidence_z: float,
) -> float:
    """Delta-method uncertainty for the threshold-crossing cycle.

    Unlike residual/slope scaling, this accounts for uncertainty in both the
    fitted slope and intercept and naturally grows for longer extrapolations.
    """
    if len(x) <= 2:
        return 0.0
    design = np.column_stack([x, np.ones(len(x))])
    fitted = slope * x + intercept
    residual = y - fitted
    dof = max(len(x) - 2, 1)
    sigma2 = float(np.sum(residual**2) / dof)
    try:
        covariance = sigma2 * np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return 0.0
    eol = float((threshold - intercept) / slope)
    gradient = np.asarray([-eol / slope, -1.0 / slope], dtype=float)
    variance = float(gradient @ covariance @ gradient)
    return float(confidence_z * np.sqrt(max(variance, 0.0)))


def estimate_capacity_eol(
    cycle_index,
    capacity_ah,
    *,
    eol_capacity_ah=1.4,
    nominal_capacity_ah=2.0,
    minimum_observations=20,
    slope_window=20,
    maximum_projection_cycles=500,
    confidence_z=1.96,
) -> dict:
    """Causal capacity-trajectory EOL estimate with a conservative interval.

    The point estimate remains the fixed recent-window linear extrapolation.
    Uncertainty combines regression-parameter uncertainty with model-form
    disagreement across fixed 20-cycle, 40-cycle and full-history regressions.
    No future observation or censoring bound is supplied to the predictor.
    """
    raw_y = np.asarray(capacity_ah, dtype=float).ravel()
    x, y = _history(cycle_index, raw_y)
    features = capacity_features(
        x, y, nominal_capacity_ah=nominal_capacity_ah, slope_window=slope_window
    )
    if not np.isfinite(eol_capacity_ah) or not 0 < eol_capacity_ah < nominal_capacity_ah:
        raise ValueError("eol_capacity_ah must lie between zero and nominal capacity")
    base = {**features, "method": "local_linear_multiscale_interval_v3",
            "missing_capacity_observations": int(np.sum(~np.isfinite(raw_y))),
            "interval_calibrated": False, "eol_state": "projected_not_observed",
            "last_observed_cycle": None if not x.size else float(x[-1])}
    if raw_y.size and not np.isfinite(raw_y[-1]):
        return {**base, "abstained": True, "abstain_reason": "latest_capacity_missing"}
    if x.size < minimum_observations:
        return {**base, "abstained": True, "abstain_reason": "insufficient_history"}

    smooth = np.asarray(features["smoothed_capacity_ah"], dtype=float)
    current_cycle = float(x[-1])
    current_capacity = float(smooth[-1])
    if current_capacity <= eol_capacity_ah:
        first = int(np.flatnonzero(smooth <= eol_capacity_ah)[0])
        crossing = float(x[first])
        lower = float(x[first-1]) if first else 0.0
        width = crossing - lower
        return {
            **base,
            "eol_state": "observed_crossing" if first else "left_censored_crossing",
            "abstained": False,
            "abstain_reason": None,
            "predicted_eol_cycle": crossing,
            "eol_interval_cycles": [lower, crossing],
            "remaining_useful_life_cycles": 0.0,
            "uncertainty_cycles": width,
            "risk": 1.0,
            "candidate_eol_cycles": [crossing],
            "parameter_uncertainty_cycles": 0.0,
            "model_disagreement_cycles": 0.0,
        }

    n = min(max(3, int(slope_window)), x.size)
    xl, yl = x[-n:], smooth[-n:]
    primary = _fit_crossing(xl, yl, float(eol_capacity_ah))
    if primary is None:
        return {**base, "abstained": True, "abstain_reason": "non_degrading_or_unstable_slope"}
    slope, intercept, eol = primary
    rul = float(eol - current_cycle)
    if rul < 0 or rul > maximum_projection_cycles:
        return {**base, "abstained": True, "abstain_reason": "projection_outside_credible_horizon"}

    parameter_uncertainty = _crossing_parameter_uncertainty(
        # Smoothing stabilizes the point fit but must not erase measurement noise
        # from the interval calculation. This remains uncalibrated uncertainty.
        xl, y[-n:], slope, intercept, float(eol_capacity_ah), float(confidence_z)
    )

    candidate_eols: list[float] = []
    for width in sorted({min(20, len(x)), min(40, len(x)), len(x)}):
        if width < 3:
            continue
        candidate = _fit_crossing(x[-width:], smooth[-width:], float(eol_capacity_ah))
        if candidate is None:
            continue
        candidate_eol = float(candidate[2])
        candidate_rul = candidate_eol - current_cycle
        if 0.0 <= candidate_rul <= float(maximum_projection_cycles):
            candidate_eols.append(candidate_eol)
    if not candidate_eols:
        candidate_eols = [eol]

    candidate_min = min(candidate_eols)
    candidate_max = max(candidate_eols)
    model_disagreement = float(max(abs(eol - candidate_min), abs(candidate_max - eol)))

    lower_eol = max(current_cycle, min(eol - parameter_uncertainty, candidate_min))
    upper_eol = max(eol + parameter_uncertainty, candidate_max)
    uncertainty = float(max(eol - lower_eol, upper_eol - eol))

    margin = max(current_capacity - eol_capacity_ah, 0.0)
    full_margin = max(nominal_capacity_ah - eol_capacity_ah, 1e-6)
    risk = float(np.clip(1.0 - margin / full_margin, 0.0, 1.0))
    return {
        **base,
        "abstained": False,
        "abstain_reason": None,
        "predicted_eol_cycle": eol,
        "eol_interval_cycles": [float(lower_eol), float(upper_eol)],
        "remaining_useful_life_cycles": rul,
        "uncertainty_cycles": uncertainty,
        "risk": risk,
        "candidate_eol_cycles": [float(value) for value in candidate_eols],
        "parameter_uncertainty_cycles": float(parameter_uncertainty),
        "model_disagreement_cycles": model_disagreement,
    }
