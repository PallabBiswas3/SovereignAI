from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..detectors.residual_changepoint import (
    DEFAULT_CUSUM_DRIFT,
    DEFAULT_CUSUM_HOLD_SAMPLES,
    DEFAULT_CUSUM_THRESHOLD,
    ResidualCUSUMConfig,
    residual_cusum,
)
from .wind_physics import WindPhysicsConfig, physics_consistency_check
from ..tools.wind_scada import (
    WindNormalBehaviorState,
    detect_wind_operating_regimes,
    fit_wind_normal_behavior_model,
    predict_wind_normal_behavior,
    scada_quality_check,
    wind_residual_anomaly_detection,
)


@dataclass
class WindScadaDiagnosticResult:
    anomaly_detected: bool
    affected_channels: list[str]
    confidence: float
    alarm_mask: np.ndarray
    anomaly_scores: np.ndarray
    tool_trace: list[str]
    artifacts: dict[str, Any]


class WindScadaDiagnosticPipeline:
    """Regime-aware normal-behavior baseline for Wind-SCADA data."""

    def __init__(
        self,
        *,
        n_regimes: int = 4,
        residual_threshold: float = 3.5,
        persistence: int = 3,
        cusum_drift: float = DEFAULT_CUSUM_DRIFT,
        cusum_threshold: float = DEFAULT_CUSUM_THRESHOLD,
        cusum_hold_samples: int = DEFAULT_CUSUM_HOLD_SAMPLES,
        cusum_min_channel_support: int = 2,
        physics_config: WindPhysicsConfig | None = None,
        random_state: int = 0,
    ):
        self.n_regimes = int(n_regimes)
        self.residual_threshold = float(residual_threshold)
        self.persistence = int(persistence)
        self.cusum_config = ResidualCUSUMConfig(
            drift=float(cusum_drift),
            threshold=float(cusum_threshold),
            hold_samples=int(cusum_hold_samples),
            min_channel_support=int(cusum_min_channel_support),
        )
        self.physics_config = physics_config or WindPhysicsConfig()
        self.random_state = int(random_state)

    @staticmethod
    def _first_matching_index(
        channel_names: list[str],
        token_groups: Iterable[Iterable[str]],
        *,
        excluded_tokens: Iterable[str] = (),
        used: set[int] | None = None,
    ) -> int | None:
        lowered = [str(name).lower() for name in channel_names]
        excluded = tuple(str(token).lower() for token in excluded_tokens)
        occupied = set() if used is None else set(used)
        for tokens in token_groups:
            required = tuple(str(token).lower() for token in tokens)
            for i, name in enumerate(lowered):
                if i in occupied:
                    continue
                if excluded and any(token in name for token in excluded):
                    continue
                if all(token in name for token in required):
                    return i
        return None

    @classmethod
    def _driver_selection(
        cls,
        channel_names: list[str],
        matrix: np.ndarray,
        max_drivers: int = 4,
    ) -> tuple[list[int], dict[str, int]]:
        """Resolve physically meaningful operating-regime drivers.

        The earlier generic ``"power"`` match could select reactive or apparent
        power before generated active power. That changes regime geometry without
        representing turbine operating load. Driver resolution is now role-aware,
        excludes reactive/apparent/power-factor channels, and records the chosen
        semantic role for auditability.
        """
        used: set[int] = set()
        roles: dict[str, int] = {}
        selected: list[int] = []

        role_specs = (
            (
                "wind_speed",
                (("wind", "speed"),),
                (),
            ),
            (
                "active_power",
                (
                    ("active", "power"),
                    ("real", "power"),
                    ("electrical", "power"),
                    ("generator", "power"),
                    ("power",),
                ),
                ("reactive", "apparent", "power factor", "power_factor", "cosphi"),
            ),
            (
                "rotor_speed",
                (("rotor", "speed"), ("rotation", "speed"), ("rotor", "rpm")),
                (),
            ),
            (
                "generator_speed",
                (("generator", "speed"), ("generator", "rpm")),
                (),
            ),
        )

        for role, token_groups, excluded in role_specs:
            idx = cls._first_matching_index(
                channel_names,
                token_groups,
                excluded_tokens=excluded,
                used=used,
            )
            if idx is None:
                continue
            selected.append(int(idx))
            roles[role] = int(idx)
            used.add(int(idx))
            if len(selected) >= min(max_drivers, matrix.shape[1]):
                break

        variances = np.nanvar(matrix, axis=0)
        lowered = [str(name).lower() for name in channel_names]
        unsafe_fallback_tokens = ("reactive", "apparent", "power factor", "power_factor", "cosphi")

        # Fill missing roles using informative channels, but do not silently
        # re-introduce known non-load electrical quantities as regime drivers.
        fallback_rank = 0
        for idx in np.argsort(variances)[::-1]:
            idx = int(idx)
            if idx in used:
                continue
            if any(token in lowered[idx] for token in unsafe_fallback_tokens):
                continue
            if not np.isfinite(variances[idx]) or variances[idx] <= 1e-12:
                continue
            selected.append(idx)
            roles[f"variance_fallback_{fallback_rank}"] = idx
            fallback_rank += 1
            used.add(idx)
            if len(selected) >= min(max_drivers, matrix.shape[1]):
                break

        # Degenerate small fixtures may contain only excluded/constant channels.
        # Preserve a usable model rather than failing, while keeping the fallback
        # explicit in diagnostics.
        if len(selected) < min(max_drivers, matrix.shape[1]):
            for idx in np.argsort(variances)[::-1]:
                idx = int(idx)
                if idx in used:
                    continue
                if not np.isfinite(variances[idx]) or variances[idx] <= 1e-12:
                    continue
                selected.append(idx)
                roles[f"last_resort_variance_{fallback_rank}"] = idx
                fallback_rank += 1
                used.add(idx)
                if len(selected) >= min(max_drivers, matrix.shape[1]):
                    break

        if not selected:
            raise ValueError("unable to resolve any non-constant operating-regime driver")
        return selected[:max_drivers], roles

    @classmethod
    def _driver_indices(cls, channel_names: list[str], matrix: np.ndarray) -> list[int]:
        drivers, _ = cls._driver_selection(channel_names, matrix)
        return drivers

    @staticmethod
    def _target_indices(channel_names: list[str], matrix: np.ndarray, max_targets: int = 8) -> list[int]:
        keywords = (
            "temp", "temperature", "bearing", "gear", "generator", "converter",
            "oil", "vibration", "pitch", "voltage", "current", "power",
        )
        selected = [i for i, name in enumerate(channel_names) if any(k in name.lower() for k in keywords)]
        selected = selected[:max_targets]
        if len(selected) < min(3, matrix.shape[1]):
            variances = np.nanvar(matrix, axis=0)
            for idx in np.argsort(variances)[::-1]:
                idx = int(idx)
                if idx not in selected and variances[idx] > 1e-12:
                    selected.append(idx)
                if len(selected) >= min(max_targets, matrix.shape[1]):
                    break
        return selected

    @staticmethod
    def _predictor_indices(matrix: np.ndarray, drivers: list[int], targets: list[int], max_predictors: int = 16) -> list[int]:
        """Bound NBM dimensionality for CARE farms with hundreds of channels."""
        variances = np.nanvar(matrix, axis=0)
        selected = []
        for idx in list(drivers) + list(targets):
            if idx not in selected and np.isfinite(variances[idx]) and variances[idx] > 1e-12:
                selected.append(int(idx))
        for idx in np.argsort(variances)[::-1]:
            idx = int(idx)
            if idx not in selected and np.isfinite(variances[idx]) and variances[idx] > 1e-12:
                selected.append(idx)
            if len(selected) >= min(max_predictors, matrix.shape[1]):
                break
        return selected[:max_predictors]

    @staticmethod
    def _fill_reference_statistics(reference: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ref = np.asarray(reference, dtype=float).copy()
        cur = np.asarray(current, dtype=float).copy()
        med = np.nanmedian(ref, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        ref_missing = ~np.isfinite(ref)
        cur_missing = ~np.isfinite(cur)
        if np.any(ref_missing):
            ref[ref_missing] = np.take(med, np.where(ref_missing)[1])
        if np.any(cur_missing):
            cur[cur_missing] = np.take(med, np.where(cur_missing)[1])
        return ref, cur

    @staticmethod
    def _regime_assignment_diagnostics(
        z_ref: np.ndarray,
        reference_regimes: np.ndarray,
        z_pred: np.ndarray,
        unique_regimes: list[int],
        centers: list[np.ndarray],
        distances: np.ndarray,
    ) -> dict[str, Any]:
        """Quantify how far deployment samples lie from healthy regime support."""
        label_to_position = {int(label): pos for pos, label in enumerate(unique_regimes)}
        reference_assigned_distance = np.asarray(
            [
                np.linalg.norm(z_ref[i] - centers[label_to_position[int(reference_regimes[i])]])
                for i in range(len(z_ref))
            ],
            dtype=float,
        )
        finite_ref = reference_assigned_distance[np.isfinite(reference_assigned_distance)]
        distance_threshold = float(np.nanpercentile(finite_ref, 99.0)) if finite_ref.size else float("inf")

        if distances.size:
            nearest = np.min(distances, axis=1)
            if distances.shape[1] > 1:
                ordered = np.sort(distances, axis=1)
                second = ordered[:, 1]
                margin = second - ordered[:, 0]
            else:
                margin = np.full(len(z_pred), np.inf)
        else:
            nearest = np.zeros(len(z_pred), dtype=float)
            margin = np.full(len(z_pred), np.inf)

        ood = nearest > distance_threshold
        return {
            "reference_distance_threshold_p99": distance_threshold,
            "prediction_nearest_distance": nearest,
            "prediction_assignment_margin": margin,
            "out_of_distribution_mask": ood,
            "out_of_distribution_fraction": float(np.mean(ood)) if len(ood) else 0.0,
            "regime_labels": list(unique_regimes),
        }

    def run(
        self,
        train_matrix,
        prediction_matrix,
        channel_names: Iterable[str],
        *,
        train_timestamps=None,
        prediction_timestamps=None,
        healthy_train_mask=None,
        operating_regime_labels=None,
        target_indices: Iterable[int] | None = None,
    ) -> WindScadaDiagnosticResult:
        train = np.asarray(train_matrix, dtype=float)
        pred = np.asarray(prediction_matrix, dtype=float)
        if train.ndim != 2 or pred.ndim != 2 or train.shape[1] != pred.shape[1]:
            raise ValueError("train/prediction matrices must be aligned 2-D arrays")
        names = [str(x) for x in channel_names]
        if len(names) != train.shape[1]:
            raise ValueError("channel_names length mismatch")

        tool_trace = ["scada_quality_check"]
        quality_train = scada_quality_check(train, names, train_timestamps)
        quality_prediction = scada_quality_check(pred, names, prediction_timestamps)

        if healthy_train_mask is not None:
            mask = np.asarray(healthy_train_mask, dtype=bool)
            if mask.shape[0] != train.shape[0]:
                raise ValueError("healthy_train_mask length mismatch")
            healthy = train[mask]
        else:
            healthy = train
        if healthy.shape[0] < 60:
            raise ValueError("need at least 60 healthy SCADA samples for normal-behavior modeling")

        healthy, pred_clean = self._fill_reference_statistics(healthy, pred)
        drivers, driver_roles = self._driver_selection(names, healthy)

        tool_trace.append("operating_regime_detection")
        reference_regimes = detect_wind_operating_regimes(
            healthy,
            driver_indices=drivers,
            operating_regime_labels=operating_regime_labels,
            n_regimes=self.n_regimes,
            random_state=self.random_state,
        )["regime_ids"]

        unique_regimes = sorted(int(v) for v in np.unique(reference_regimes))
        driver_ref = healthy[:, drivers]
        driver_med = np.nanmedian(driver_ref, axis=0)
        driver_scale = np.nanmedian(np.abs(driver_ref - driver_med), axis=0) * 1.4826
        driver_scale = np.where(driver_scale < 1e-12, np.nanstd(driver_ref, axis=0), driver_scale)
        driver_scale = np.where(driver_scale < 1e-12, 1.0, driver_scale)
        z_ref = (driver_ref - driver_med) / driver_scale
        centers = [np.mean(z_ref[reference_regimes == regime], axis=0) for regime in unique_regimes]
        z_pred = (pred_clean[:, drivers] - driver_med) / driver_scale
        distances = np.stack([np.linalg.norm(z_pred - center, axis=1) for center in centers], axis=1)
        prediction_regimes = np.asarray([unique_regimes[i] for i in np.argmin(distances, axis=1)])
        regime_diagnostics = self._regime_assignment_diagnostics(
            z_ref,
            reference_regimes,
            z_pred,
            unique_regimes,
            centers,
            distances,
        )

        targets = self._target_indices(names, healthy) if target_indices is None else [int(i) for i in target_indices]
        predictors = self._predictor_indices(healthy, drivers, targets)

        tool_trace.append("normal_behavior_model")
        state: WindNormalBehaviorState = fit_wind_normal_behavior_model(
            healthy,
            reference_regimes,
            target_indices=targets,
            predictor_indices=predictors,
            max_train_samples_per_regime=5000,
            random_state=self.random_state,
        )
        predicted = predict_wind_normal_behavior(pred_clean, prediction_regimes, state)

        tool_trace.extend(["residual_analysis", "scada_anomaly_detection"])
        anomaly = wind_residual_anomaly_detection(
            predicted["normalized_residuals"],
            threshold=self.residual_threshold,
            persistence=self.persistence,
        )

        # Regime switches are known operating-point boundaries, not evidence of
        # a persistent fault. Clear CUSUM memory at the first sample of each new
        # regime to prevent normal transitions from integrating into alarms.
        regime_transition = np.zeros(len(prediction_regimes), dtype=bool)
        if len(prediction_regimes) > 1:
            regime_transition[1:] = prediction_regimes[1:] != prediction_regimes[:-1]

        tool_trace.append("residual_cusum")
        changepoint = residual_cusum(
            predicted["normalized_residuals"],
            timestamps=prediction_timestamps,
            reset_mask=regime_transition,
            config=self.cusum_config,
        )

        tool_trace.append("physics_consistency_check")
        physics = physics_consistency_check(
            pred_clean,
            names,
            normal_reference=healthy,
            config=self.physics_config,
        )

        residual_alarm = np.asarray(anomaly["alarm_mask"], dtype=bool)
        drift_alarm = np.asarray(changepoint["alarm_mask"], dtype=bool)
        fused_alarm = residual_alarm | drift_alarm
        residual_score = np.asarray(anomaly["anomaly_scores"], dtype=float)
        drift_score = np.asarray(changepoint["cusum_scores"], dtype=float) / max(self.cusum_config.threshold, 1e-12)
        score = np.maximum(residual_score / max(self.residual_threshold, 1e-12), drift_score)

        channel_strength = np.nanpercentile(np.abs(predicted["normalized_residuals"]), 95, axis=0)
        order = np.argsort(channel_strength)[::-1]
        affected = [names[targets[int(i)]] for i in order[: min(5, len(order))] if channel_strength[int(i)] >= 1.0]
        for finding in physics["verification_findings"]:
            channel = finding.get("channel")
            if channel and channel not in affected:
                affected.append(str(channel))
        affected = affected[:8]

        physics_boost = min(0.15, 0.03 * len(physics["verification_findings"]))
        confidence = float(np.clip(np.nanpercentile(score, 95) / 2.0 + physics_boost, 0.0, 1.0))

        driver_role_details = {
            role: {"index": int(idx), "channel": names[int(idx)]}
            for role, idx in driver_roles.items()
        }

        return WindScadaDiagnosticResult(
            anomaly_detected=bool(np.any(fused_alarm)),
            affected_channels=affected,
            confidence=confidence,
            alarm_mask=fused_alarm,
            anomaly_scores=score,
            tool_trace=tool_trace,
            artifacts={
                "quality_train": quality_train,
                "quality_prediction": quality_prediction,
                "driver_indices": drivers,
                "driver_channel_names": [names[int(i)] for i in drivers],
                "driver_roles": driver_role_details,
                "regime_assignment": regime_diagnostics,
                "target_indices": targets,
                "predictor_indices": predictors,
                "reference_regimes": reference_regimes,
                "prediction_regimes": prediction_regimes,
                "regime_transition_mask": regime_transition,
                "normal_behavior": predicted,
                "anomaly_detection": anomaly,
                "residual_changepoint": changepoint,
                "physics_consistency": physics,
                "fusion": {
                    "residual_alarm_fraction": float(np.mean(residual_alarm)),
                    "drift_alarm_fraction": float(np.mean(drift_alarm)),
                    "fused_alarm_fraction": float(np.mean(fused_alarm)),
                },
            },
        )
