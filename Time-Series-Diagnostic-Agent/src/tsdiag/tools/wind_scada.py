from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("expected [samples, channels]")
    return arr


def scada_quality_check(signal_matrix, channel_names, timestamps=None) -> dict[str, Any]:
    x = _as_2d(signal_matrix)
    names = [str(v) for v in channel_names]
    if len(names) != x.shape[1]:
        raise ValueError("channel_names length mismatch")

    flags = []
    missingness = {}
    for i, name in enumerate(names):
        col = x[:, i]
        missing = float(np.mean(~np.isfinite(col)))
        missingness[name] = missing
        finite = col[np.isfinite(col)]
        if missing > 0:
            flags.append({"channel": name, "flag": "missing_values", "fraction": missing})
        if finite.size and np.nanstd(finite) < 1e-12:
            flags.append({"channel": name, "flag": "constant_channel"})

    irregularity = None
    if timestamps is not None:
        ts = np.asarray(timestamps)
        if ts.shape[0] != x.shape[0]:
            raise ValueError("timestamps length mismatch")
        try:
            t = ts.astype("datetime64[ns]").astype("int64")
            dt = np.diff(t).astype(float)
            if dt.size:
                median = np.median(dt)
                irregularity = float(np.mean(np.abs(dt - median) > max(abs(median) * 0.05, 1.0)))
                if irregularity > 0.01:
                    flags.append({"flag": "irregular_sampling", "fraction": irregularity})
        except Exception:
            flags.append({"flag": "unparseable_timestamps"})

    return {
        "quality_flags": flags,
        "missingness": missingness,
        "sampling_irregularity": irregularity,
    }


def detect_wind_operating_regimes(
    signal_matrix,
    channel_names=None,
    *,
    driver_indices: Iterable[int] | None = None,
    operating_regime_labels=None,
    n_regimes: int = 4,
    random_state: int = 0,
) -> dict[str, Any]:
    """Cluster healthy operation into comparable regimes."""
    x = _as_2d(signal_matrix)
    if channel_names is not None and len(channel_names) != x.shape[1]:
        raise ValueError("channel_names length mismatch")
    if operating_regime_labels is not None:
        labels = np.asarray(operating_regime_labels)
        if labels.shape[0] != x.shape[0]:
            raise ValueError("operating_regime_labels length mismatch")
    else:
        if driver_indices is None:
            variances = np.nanvar(x, axis=0)
            driver_indices = np.argsort(variances)[::-1][: min(4, x.shape[1])]
        cols = np.asarray(list(driver_indices), dtype=int)
        if cols.size == 0:
            raise ValueError("at least one regime driver is required")
        drivers = x[:, cols]
        med = np.nanmedian(drivers, axis=0)
        scale = np.nanmedian(np.abs(drivers - med), axis=0) * 1.4826
        scale = np.where(scale < 1e-12, np.nanstd(drivers, axis=0), scale)
        scale = np.where(scale < 1e-12, 1.0, scale)
        z = np.nan_to_num((drivers - med) / scale)
        k = max(1, min(int(n_regimes), len(x)))
        labels = KMeans(n_clusters=k, n_init="auto", random_state=random_state).fit_predict(z)

    descriptions = {}
    for value in np.unique(labels):
        mask = labels == value
        descriptions[int(value)] = {
            "samples": int(np.sum(mask)),
            "mean": np.nanmean(x[mask], axis=0).tolist(),
        }
    return {"regime_ids": labels, "regime_descriptions": descriptions}


@dataclass
class WindNormalBehaviorState:
    target_indices: list[int]
    predictor_indices: list[int]
    models: dict[tuple[int, int], HistGradientBoostingRegressor]
    global_models: dict[int, HistGradientBoostingRegressor]
    residual_center: np.ndarray
    residual_scale: np.ndarray
    residual_center_by_regime: dict[int, np.ndarray]
    residual_scale_by_regime: dict[int, np.ndarray]
    regime_ids: list[int]


def _valid_rows(x: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(x), axis=1)


def _new_nbm(random_state: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=random_state,
    )


def _subsample_indices(idx: np.ndarray, maximum: int) -> np.ndarray:
    if idx.size <= maximum:
        return idx
    positions = np.linspace(0, idx.size - 1, maximum).astype(int)
    return idx[positions]


def _robust_center_scale(values: np.ndarray, *, minimum_samples: int = 8) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < int(minimum_samples):
        return float("nan"), float("nan")
    center = float(np.nanmedian(values))
    scale = float(np.nanmedian(np.abs(values - center)) * 1.4826)
    if not np.isfinite(scale) or scale < 1e-12:
        scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale < 1e-12:
        return center, float("nan")
    return center, scale


def _oos_residual_calibration(
    ref: np.ndarray,
    regimes: np.ndarray,
    target: int,
    feature_cols: list[int],
    valid: np.ndarray,
    *,
    max_train_samples_per_regime: int,
    random_state: int,
) -> tuple[float, float, dict[int, float], dict[int, float]]:
    """Estimate global and regime-specific healthy residual calibration.

    A deterministic stratified 20% holdout estimates out-of-sample prediction
    residuals. The global center/scale remain as robust fallbacks, while each
    sufficiently represented operating regime receives its own center and scale.
    This prevents normal regime-specific model bias or heteroscedasticity from
    being interpreted as a persistent fault by residual thresholding/CUSUM.
    """
    fit_mask = np.zeros(ref.shape[0], dtype=bool)
    cal_mask = np.zeros(ref.shape[0], dtype=bool)
    for regime in np.unique(regimes[valid]):
        idx = np.flatnonzero(valid & (regimes == regime))
        if idx.size < 10:
            continue
        cal_idx = idx[::5]
        fit_idx = np.setdiff1d(idx, cal_idx, assume_unique=True)
        cal_mask[cal_idx] = True
        fit_mask[fit_idx] = True

    if np.sum(cal_mask) < 10 or np.sum(fit_mask) < 20:
        return float("nan"), float("nan"), {}, {}

    global_fit = np.flatnonzero(fit_mask)
    global_model = _new_nbm(random_state)
    global_model.fit(ref[global_fit][:, feature_cols], ref[global_fit, target])

    expected = np.full(ref.shape[0], np.nan)
    for regime in np.unique(regimes[cal_mask]):
        cal_idx = np.flatnonzero(cal_mask & (regimes == regime))
        fit_idx = np.flatnonzero(fit_mask & (regimes == regime))
        if fit_idx.size >= 30:
            fit_idx = _subsample_indices(fit_idx, max_train_samples_per_regime)
            model = _new_nbm(random_state)
            model.fit(ref[fit_idx][:, feature_cols], ref[fit_idx, target])
        else:
            model = global_model
        expected[cal_idx] = model.predict(ref[cal_idx][:, feature_cols])

    residual = ref[:, target] - expected
    center, scale = _robust_center_scale(residual[cal_mask], minimum_samples=10)

    by_regime_center: dict[int, float] = {}
    by_regime_scale: dict[int, float] = {}
    for regime in np.unique(regimes[cal_mask]):
        regime = int(regime)
        regime_residual = residual[cal_mask & (regimes == regime)]
        regime_center, regime_scale = _robust_center_scale(regime_residual, minimum_samples=8)
        if np.isfinite(regime_center):
            by_regime_center[regime] = float(regime_center)
        if np.isfinite(regime_scale) and regime_scale >= 1e-12:
            by_regime_scale[regime] = float(regime_scale)

    return center, scale, by_regime_center, by_regime_scale


def fit_wind_normal_behavior_model(
    normal_reference,
    reference_regime_ids,
    *,
    target_indices: Iterable[int] | None = None,
    predictor_indices: Iterable[int] | None = None,
    max_train_samples_per_regime: int = 20000,
    random_state: int = 0,
) -> WindNormalBehaviorState:
    ref = _as_2d(normal_reference)
    regimes = np.asarray(reference_regime_ids)
    if regimes.shape[0] != ref.shape[0]:
        raise ValueError("reference_regime_ids length mismatch")

    targets = list(range(ref.shape[1])) if target_indices is None else [int(i) for i in target_indices]
    predictors = list(range(ref.shape[1])) if predictor_indices is None else [int(i) for i in predictor_indices]
    if not targets or not predictors:
        raise ValueError("normal behavior model requires targets and predictors")

    regime_ids = [int(v) for v in np.unique(regimes)]
    models: dict[tuple[int, int], HistGradientBoostingRegressor] = {}
    global_models: dict[int, HistGradientBoostingRegressor] = {}
    expected_ref = np.full((ref.shape[0], len(targets)), np.nan)
    centers = np.full(len(targets), np.nan)
    scales = np.full(len(targets), np.nan)
    regime_centers = {regime: np.full(len(targets), np.nan) for regime in regime_ids}
    regime_scales = {regime: np.full(len(targets), np.nan) for regime in regime_ids}

    for target_pos, target in enumerate(targets):
        feature_cols = [j for j in predictors if j != target]
        if not feature_cols:
            feature_cols = predictors
        valid = _valid_rows(ref[:, feature_cols]) & np.isfinite(ref[:, target])
        if np.sum(valid) < 10:
            raise ValueError(f"insufficient finite reference samples for target {target}")

        center, scale, center_by_regime, scale_by_regime = _oos_residual_calibration(
            ref,
            regimes,
            target,
            feature_cols,
            valid,
            max_train_samples_per_regime=max_train_samples_per_regime,
            random_state=random_state,
        )
        centers[target_pos] = center
        scales[target_pos] = scale
        for regime, value in center_by_regime.items():
            if regime in regime_centers:
                regime_centers[regime][target_pos] = value
        for regime, value in scale_by_regime.items():
            if regime in regime_scales:
                regime_scales[regime][target_pos] = value

        global_model = _new_nbm(random_state)
        global_model.fit(ref[valid][:, feature_cols], ref[valid, target])
        global_models[target] = global_model

        for regime in regime_ids:
            idx = np.flatnonzero(valid & (regimes == regime))
            if idx.size < 30:
                continue
            idx = _subsample_indices(idx, max_train_samples_per_regime)
            model = _new_nbm(random_state)
            model.fit(ref[idx][:, feature_cols], ref[idx, target])
            models[(regime, target)] = model

        for regime in regime_ids:
            mask = regimes == regime
            valid_mask = mask & _valid_rows(ref[:, feature_cols])
            if not np.any(valid_mask):
                continue
            model = models.get((regime, target), global_model)
            expected_ref[valid_mask, target_pos] = model.predict(ref[valid_mask][:, feature_cols])

    in_sample_residuals = ref[:, targets] - expected_ref
    fallback_center = np.nanmedian(in_sample_residuals, axis=0)
    fallback_scale = np.nanmedian(
        np.abs(in_sample_residuals - fallback_center), axis=0
    ) * 1.4826
    fallback_std = np.nanstd(in_sample_residuals, axis=0)
    fallback_scale = np.where(fallback_scale < 1e-12, fallback_std, fallback_scale)

    residual_center = np.where(np.isfinite(centers), centers, fallback_center)
    residual_center = np.where(np.isfinite(residual_center), residual_center, 0.0)
    residual_scale = np.where(np.isfinite(scales) & (scales >= 1e-12), scales, fallback_scale)
    residual_scale = np.where(np.isfinite(residual_scale) & (residual_scale >= 1e-12), residual_scale, 1.0)

    for regime in regime_ids:
        regime_mask = regimes == regime
        regime_residuals = in_sample_residuals[regime_mask]
        in_center = np.nanmedian(regime_residuals, axis=0) if np.any(regime_mask) else residual_center
        in_scale = (
            np.nanmedian(np.abs(regime_residuals - in_center), axis=0) * 1.4826
            if np.any(regime_mask)
            else residual_scale
        )
        in_std = np.nanstd(regime_residuals, axis=0) if np.any(regime_mask) else residual_scale
        in_scale = np.where(np.isfinite(in_scale) & (in_scale >= 1e-12), in_scale, in_std)

        regime_centers[regime] = np.where(
            np.isfinite(regime_centers[regime]), regime_centers[regime], in_center
        )
        regime_centers[regime] = np.where(
            np.isfinite(regime_centers[regime]), regime_centers[regime], residual_center
        )
        regime_scales[regime] = np.where(
            np.isfinite(regime_scales[regime]) & (regime_scales[regime] >= 1e-12),
            regime_scales[regime],
            in_scale,
        )
        regime_scales[regime] = np.where(
            np.isfinite(regime_scales[regime]) & (regime_scales[regime] >= 1e-12),
            regime_scales[regime],
            residual_scale,
        )

    return WindNormalBehaviorState(
        target_indices=targets,
        predictor_indices=predictors,
        models=models,
        global_models=global_models,
        residual_center=np.asarray(residual_center, dtype=float),
        residual_scale=np.asarray(residual_scale, dtype=float),
        residual_center_by_regime={k: np.asarray(v, dtype=float) for k, v in regime_centers.items()},
        residual_scale_by_regime={k: np.asarray(v, dtype=float) for k, v in regime_scales.items()},
        regime_ids=regime_ids,
    )


def predict_wind_normal_behavior(signal_matrix, regime_ids, state: WindNormalBehaviorState) -> dict[str, Any]:
    x = _as_2d(signal_matrix)
    regimes = np.asarray(regime_ids)
    if regimes.shape[0] != x.shape[0]:
        raise ValueError("regime_ids length mismatch")

    expected = np.full((x.shape[0], len(state.target_indices)), np.nan)
    for target_pos, target in enumerate(state.target_indices):
        feature_cols = [j for j in state.predictor_indices if j != target]
        if not feature_cols:
            feature_cols = state.predictor_indices
        valid_features = _valid_rows(x[:, feature_cols])
        for regime in np.unique(regimes):
            mask = (regimes == regime) & valid_features
            if not np.any(mask):
                continue
            model = state.models.get((int(regime), target), state.global_models[target])
            expected[mask, target_pos] = model.predict(x[mask][:, feature_cols])

    observed = x[:, state.target_indices]
    residuals = observed - expected

    center_used = np.tile(state.residual_center, (x.shape[0], 1))
    scale_used = np.tile(state.residual_scale, (x.shape[0], 1))
    for regime in np.unique(regimes):
        regime = int(regime)
        mask = regimes == regime
        if regime in state.residual_center_by_regime:
            center_used[mask] = state.residual_center_by_regime[regime]
        if regime in state.residual_scale_by_regime:
            scale_used[mask] = state.residual_scale_by_regime[regime]
    scale_used = np.where(np.isfinite(scale_used) & (scale_used >= 1e-12), scale_used, 1.0)

    normalized = (residuals - center_used) / scale_used
    return {
        "expected_signal": expected,
        "observed_signal": observed,
        "residuals": residuals,
        "residual_center": np.asarray(state.residual_center, dtype=float),
        "residual_scale": np.asarray(state.residual_scale, dtype=float),
        "residual_center_by_regime": {
            int(k): np.asarray(v, dtype=float) for k, v in state.residual_center_by_regime.items()
        },
        "residual_scale_by_regime": {
            int(k): np.asarray(v, dtype=float) for k, v in state.residual_scale_by_regime.items()
        },
        "residual_center_used": center_used,
        "residual_scale_used": scale_used,
        "normalized_residuals": normalized,
        "model_uncertainty": scale_used,
        "target_indices": list(state.target_indices),
    }


def normal_behavior_model(
    signal_matrix,
    regime_ids,
    *,
    normal_reference=None,
    reference_regime_ids=None,
    target_indices: Iterable[int] | None = None,
    predictor_indices: Iterable[int] | None = None,
    physics_model=None,
    ambient_conditions=None,
) -> dict[str, Any]:
    x = _as_2d(signal_matrix)
    ref = x if normal_reference is None else _as_2d(normal_reference)
    ref_regimes = np.asarray(regime_ids if reference_regime_ids is None else reference_regime_ids)
    state = fit_wind_normal_behavior_model(
        ref,
        ref_regimes,
        target_indices=target_indices,
        predictor_indices=predictor_indices,
    )
    prediction = predict_wind_normal_behavior(x, regime_ids, state)
    return {
        "expected_signal": prediction["expected_signal"],
        "model_uncertainty": prediction["model_uncertainty"],
        "normal_behavior_state": state,
        "residual_center": prediction["residual_center"],
        "residual_scale": prediction["residual_scale"],
        "residual_center_by_regime": prediction["residual_center_by_regime"],
        "residual_scale_by_regime": prediction["residual_scale_by_regime"],
        "normalized_residuals": prediction["normalized_residuals"],
    }


def wind_residual_anomaly_detection(
    normalized_residuals,
    *,
    threshold: float = 3.5,
    persistence: int = 3,
) -> dict[str, Any]:
    """Detect persistent large residuals without cross-channel persistence leakage.

    Persistence is evaluated independently per channel. The aggregate alarm is
    then the union of channels that individually satisfied the persistence rule.
    This prevents alternating one-sample spikes on different channels from being
    misclassified as one persistent anomaly episode.
    """
    z = np.abs(_as_2d(normalized_residuals))
    per_channel = z >= float(threshold)
    score = np.nanmax(z, axis=1)

    k = max(1, int(persistence))
    persistent_per_channel = np.zeros_like(per_channel, dtype=bool)
    runs = np.zeros(per_channel.shape[1], dtype=int)
    for i in range(per_channel.shape[0]):
        runs = np.where(per_channel[i], runs + 1, 0)
        persistent_per_channel[i] = runs >= k

    raw_alarm = np.any(per_channel, axis=1)
    alarm = np.any(persistent_per_channel, axis=1)

    return {
        "anomaly_scores": score,
        "channel_alarm_mask": per_channel,
        "persistent_channel_alarm_mask": persistent_per_channel,
        "raw_alarm_mask": raw_alarm,
        "alarm_mask": alarm,
        "threshold": float(threshold),
        "persistence": k,
    }
