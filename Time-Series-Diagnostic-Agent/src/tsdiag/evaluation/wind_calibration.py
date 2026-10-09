from __future__ import annotations

"""Healthy-only target-FAR calibration for the Wind-SCADA alarm chain."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..detectors.residual_changepoint import ResidualCUSUMConfig, residual_cusum
from ..domains.wind_scada_runner import WindScadaDiagnosticPipeline
from .split_manifest import DevelopmentSplitManifest, SplitUnit
from .wind_scada import wind_event_evidence_decision
from ..tools.wind_scada import wind_residual_anomaly_detection


WIND_CALIBRATION_SCHEMA = "tsdiag.wind-calibration"
WIND_CALIBRATION_VERSION = "1.1"
DEFAULT_WIND_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "wind_calibrated_config.json"
)


@dataclass(frozen=True)
class WindCalibrationCandidate:
    residual_threshold: float
    persistence: int
    cusum_drift: float
    cusum_threshold: float
    event_minimum_residual_fraction: float


@dataclass(frozen=True)
class WindCalibrationResult:
    config: WindCalibrationCandidate
    achieved_event_far: float
    achieved_event_far_ci: tuple[float, float]
    achieved_sample_far: float
    target_event_far: float
    n_holdout_samples: int
    n_events_used: int
    event_ids: tuple[int, ...]
    development_pool_id: str
    evaluation_pool_id: str
    leakage_boundary: str
    split_manifest_sha256: str
    met_target: bool
    grid: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": WIND_CALIBRATION_SCHEMA,
            "schema_version": WIND_CALIBRATION_VERSION,
            "development_pool_id": self.development_pool_id,
            "evaluation_pool_id": self.evaluation_pool_id,
            "leakage_boundary": self.leakage_boundary,
            "split_manifest_sha256": self.split_manifest_sha256,
            "event_ids": list(self.event_ids),
            "config": asdict(self.config),
            "target_event_far": self.target_event_far,
            "achieved_event_far": self.achieved_event_far,
            "achieved_event_far_ci": list(self.achieved_event_far_ci),
            "achieved_sample_far": self.achieved_sample_far,
            "n_holdout_samples": self.n_holdout_samples,
            "n_events_used": self.n_events_used,
            "met_target": self.met_target,
            "grid": self.grid,
        }


@dataclass(frozen=True)
class _PreparedHealthyEvent:
    event_id: int
    normalized_residuals: np.ndarray
    regime_transition_mask: np.ndarray


def _present_identifier(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def wind_split_unit(event: Mapping[str, Any]) -> SplitUnit:
    """Build a leakage unit from CARE event metadata.

    ``asset_ids`` should contain identities recovered from event rows when the
    event-info table does not expose a usable asset ID.
    """

    event_id = str(int(event["event_id"]))
    wind_farm = _present_identifier(event.get("wind_farm"))
    asset_values = event.get("asset_ids")
    if asset_values is None:
        asset_values = (event.get("asset_id"),)
    groups = []
    for value in asset_values:
        asset_id = _present_identifier(value)
        if asset_id is None:
            continue
        group = f"farm:{wind_farm or 'unknown'}/asset:{asset_id}"
        if group not in groups:
            groups.append(group)
    return SplitUnit(
        unit_id=event_id,
        group_ids=tuple(sorted(groups)),
        start_time=(
            None if event.get("event_start") is None else str(event["event_start"])
        ),
        end_time=None if event.get("event_end") is None else str(event["event_end"]),
        metadata={"wind_farm": wind_farm},
    )


def build_wind_split_manifest(
    development_events: Iterable[Mapping[str, Any]],
    evaluation_events: Iterable[Mapping[str, Any]],
    *,
    dataset_id: str,
    development_pool_id: str,
    evaluation_pool_id: str,
    leakage_boundary: str,
) -> DevelopmentSplitManifest:
    boundary_map = {"event": "unit", "asset": "group", "temporal": "temporal"}
    if leakage_boundary not in boundary_map:
        raise ValueError("Wind leakage_boundary must be event, asset, or temporal")
    manifest = DevelopmentSplitManifest(
        dataset_id=str(dataset_id),
        development_pool_id=str(development_pool_id),
        evaluation_pool_id=str(evaluation_pool_id),
        leakage_boundary=boundary_map[leakage_boundary],  # type: ignore[arg-type]
        unit_kind="care_event",
        group_kind=None if leakage_boundary == "event" else "wind_asset",
        development_units=tuple(wind_split_unit(event) for event in development_events),
        evaluation_units=tuple(wind_split_unit(event) for event in evaluation_events),
        metadata={
            "domain": "wind_scada",
            "requested_boundary": leakage_boundary,
            "asset_identity": "wind_farm + asset_id",
        },
    )
    manifest.validate()
    return manifest


def _longest_true_slice(mask: np.ndarray) -> slice:
    values = np.asarray(mask, dtype=bool).ravel()
    best_start = best_stop = start = 0
    for index, value in enumerate(values):
        if value:
            if index == 0 or not values[index - 1]:
                start = index
            if index + 1 - start > best_stop - best_start:
                best_start, best_stop = start, index + 1
    return slice(best_start, best_stop)


def _split_healthy_window(
    train_matrix: np.ndarray,
    healthy_train_mask: np.ndarray,
    *,
    holdout_fraction: float = 0.3,
    minimum_partition_samples: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Return contiguous fit/head and holdout/tail slices from healthy data."""

    matrix = np.asarray(train_matrix, dtype=float)
    mask = np.asarray(healthy_train_mask, dtype=bool).ravel()
    if matrix.ndim != 2 or mask.shape != (len(matrix),):
        raise ValueError("train_matrix and healthy_train_mask must align")
    if not 0.0 < float(holdout_fraction) < 1.0:
        raise ValueError("holdout_fraction must be between zero and one")
    healthy_slice = _longest_true_slice(mask)
    healthy = matrix[healthy_slice]
    minimum_holdout = max(10, int(minimum_partition_samples))
    minimum_fit = max(60, minimum_holdout)
    if len(healthy) < minimum_fit + minimum_holdout:
        raise ValueError(
            "event does not contain a sufficiently long contiguous healthy training window"
        )
    split_at = int(round(len(healthy) * (1.0 - float(holdout_fraction))))
    split_at = max(minimum_fit, min(split_at, len(healthy) - minimum_holdout))
    return healthy[:split_at], healthy[split_at:]


def _prepare_healthy_event(
    event: dict[str, Any],
    *,
    holdout_fraction: float,
    n_regimes: int,
    cusum_hold_samples: int,
    cusum_min_channel_support: int,
) -> _PreparedHealthyEvent:
    required = {"event_id", "train_matrix", "healthy_train_mask", "channel_names"}
    missing = sorted(required - set(event))
    if missing:
        raise ValueError(f"development event is missing fields: {missing}")
    fit, holdout = _split_healthy_window(
        event["train_matrix"],
        event["healthy_train_mask"],
        holdout_fraction=holdout_fraction,
    )
    pipeline = WindScadaDiagnosticPipeline(
        n_regimes=n_regimes,
        residual_threshold=3.5,
        persistence=3,
        cusum_drift=0.5,
        cusum_threshold=10.0,
        cusum_hold_samples=cusum_hold_samples,
        cusum_min_channel_support=cusum_min_channel_support,
    )
    result = pipeline.run(
        fit,
        holdout,
        event["channel_names"],
        healthy_train_mask=np.ones(len(fit), dtype=bool),
    )
    artifacts = result.artifacts
    return _PreparedHealthyEvent(
        event_id=int(event["event_id"]),
        normalized_residuals=np.asarray(
            artifacts["normal_behavior"]["normalized_residuals"], dtype=float
        ),
        regime_transition_mask=np.asarray(
            artifacts["regime_transition_mask"], dtype=bool
        ),
    )


def _candidate_event_metrics(
    event: _PreparedHealthyEvent,
    candidate: WindCalibrationCandidate,
    *,
    cusum_hold_samples: int,
    cusum_min_channel_support: int,
    event_minimum_corroborated_run: int,
    event_minimum_drift_fraction: float,
) -> tuple[bool, int, int]:
    residual = wind_residual_anomaly_detection(
        event.normalized_residuals,
        threshold=candidate.residual_threshold,
        persistence=candidate.persistence,
    )
    cusum = residual_cusum(
        event.normalized_residuals,
        reset_mask=event.regime_transition_mask,
        config=ResidualCUSUMConfig(
            drift=candidate.cusum_drift,
            threshold=candidate.cusum_threshold,
            hold_samples=cusum_hold_samples,
            min_channel_support=cusum_min_channel_support,
        ),
    )
    residual_mask = np.asarray(residual["alarm_mask"], dtype=bool)
    drift_mask = np.asarray(cusum["alarm_mask"], dtype=bool)
    decision = wind_event_evidence_decision(
        residual_alarm_mask=residual_mask,
        drift_alarm_mask=drift_mask,
        minimum_corroborated_run=event_minimum_corroborated_run,
        minimum_residual_fraction=candidate.event_minimum_residual_fraction,
        minimum_drift_fraction=event_minimum_drift_fraction,
    )
    fused = residual_mask | drift_mask
    return decision.event_detected, int(np.sum(fused)), int(len(fused))


def _bootstrap_interval(values: Sequence[float], *, samples: int, seed: int) -> tuple[float, float]:
    observed = np.asarray(values, dtype=float)
    if observed.size == 0:
        raise ValueError("cannot bootstrap an empty event set")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(observed), size=(int(samples), len(observed)))
    estimates = np.mean(observed[indices], axis=1)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def calibrate_wind_residual_config(
    development_events: Iterable[dict[str, Any]],
    *,
    split_manifest: DevelopmentSplitManifest,
    target_event_far: float = 0.08,
    n_regimes: int = 4,
    residual_threshold_grid: Sequence[float] = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0),
    persistence_grid: Sequence[int] = (2, 3, 4, 5),
    cusum_drift_grid: Sequence[float] = (0.5, 0.75, 1.0),
    cusum_threshold_grid: Sequence[float] = (10.0, 14.0, 18.0, 22.0),
    event_minimum_residual_fraction_grid: Sequence[float] = (0.02, 0.04, 0.06, 0.08),
    event_minimum_drift_fraction: float = 0.02,
    event_minimum_corroborated_run: int = 6,
    cusum_hold_samples: int = 6,
    cusum_min_channel_support: int = 2,
    holdout_fraction: float = 0.3,
    bootstrap_samples: int = 500,
    seed: int = 0,
) -> WindCalibrationResult:
    """Select the loosest Wind configuration meeting a healthy event-FAR target.

    The caller must supply a validated development/evaluation split manifest.
    Any unit/group/temporal leakage is rejected before models or alarm metrics
    are fit. Only each development event's healthy training window is consumed.
    """

    split_manifest.validate()
    if not 0.0 <= float(target_event_far) <= 1.0:
        raise ValueError("target_event_far must be between zero and one")
    if int(bootstrap_samples) < 1:
        raise ValueError("bootstrap_samples must be positive")
    events = list(development_events)
    if not events:
        raise ValueError("no development events supplied for Wind FAR calibration")
    event_ids = tuple(int(event["event_id"]) for event in events)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("development event IDs must be unique")
    split_manifest.assert_development_units(event_ids)

    grids = (
        residual_threshold_grid,
        persistence_grid,
        cusum_drift_grid,
        cusum_threshold_grid,
        event_minimum_residual_fraction_grid,
    )
    if any(len(values) == 0 for values in grids):
        raise ValueError("calibration grids must be non-empty")

    prepared = [
        _prepare_healthy_event(
            event,
            holdout_fraction=holdout_fraction,
            n_regimes=n_regimes,
            cusum_hold_samples=cusum_hold_samples,
            cusum_min_channel_support=cusum_min_channel_support,
        )
        for event in events
    ]

    rows: list[dict[str, Any]] = []
    candidate_index = 0
    for residual_threshold in sorted(float(v) for v in residual_threshold_grid):
        for persistence in sorted(int(v) for v in persistence_grid):
            for cusum_drift in sorted(float(v) for v in cusum_drift_grid):
                for cusum_threshold in sorted(float(v) for v in cusum_threshold_grid):
                    for minimum_residual_fraction in sorted(
                        float(v) for v in event_minimum_residual_fraction_grid
                    ):
                        candidate = WindCalibrationCandidate(
                            residual_threshold=residual_threshold,
                            persistence=persistence,
                            cusum_drift=cusum_drift,
                            cusum_threshold=cusum_threshold,
                            event_minimum_residual_fraction=minimum_residual_fraction,
                        )
                        metrics = [
                            _candidate_event_metrics(
                                event,
                                candidate,
                                cusum_hold_samples=cusum_hold_samples,
                                cusum_min_channel_support=cusum_min_channel_support,
                                event_minimum_corroborated_run=event_minimum_corroborated_run,
                                event_minimum_drift_fraction=event_minimum_drift_fraction,
                            )
                            for event in prepared
                        ]
                        flags = [float(flag) for flag, _, _ in metrics]
                        alarm_samples = sum(count for _, count, _ in metrics)
                        total_samples = sum(count for _, _, count in metrics)
                        event_far = float(np.mean(flags))
                        row = {
                            **asdict(candidate),
                            "achieved_event_far": event_far,
                            "achieved_event_far_ci": list(_bootstrap_interval(
                                flags,
                                samples=bootstrap_samples,
                                seed=seed + candidate_index,
                            )),
                            "achieved_sample_far": float(alarm_samples / max(total_samples, 1)),
                            "event_false_alarms": int(sum(flags)),
                        }
                        rows.append(row)
                        candidate_index += 1

    passing = [row for row in rows if row["achieved_event_far"] <= target_event_far]

    def preference(row: dict[str, Any]) -> tuple[float, float, int, float, float, float]:
        return (
            float(row["achieved_event_far"]),
            -float(row["residual_threshold"]),
            -int(row["persistence"]),
            -float(row["cusum_drift"]),
            -float(row["cusum_threshold"]),
            -float(row["event_minimum_residual_fraction"]),
        )

    if passing:
        chosen = max(passing, key=preference)
    else:
        chosen = min(rows, key=lambda row: (row["achieved_event_far"],) + preference(row)[1:])
    config = WindCalibrationCandidate(**{
        name: chosen[name]
        for name in WindCalibrationCandidate.__dataclass_fields__
    })
    return WindCalibrationResult(
        config=config,
        achieved_event_far=float(chosen["achieved_event_far"]),
        achieved_event_far_ci=tuple(chosen["achieved_event_far_ci"]),
        achieved_sample_far=float(chosen["achieved_sample_far"]),
        target_event_far=float(target_event_far),
        n_holdout_samples=sum(len(event.normalized_residuals) for event in prepared),
        n_events_used=len(prepared),
        event_ids=event_ids,
        development_pool_id=split_manifest.development_pool_id,
        evaluation_pool_id=split_manifest.evaluation_pool_id,
        leakage_boundary=split_manifest.leakage_boundary,
        split_manifest_sha256=split_manifest.sha256(),
        met_target=bool(passing),
        grid=rows,
    )


def save_wind_calibration_result(
    result: WindCalibrationResult,
    path: str | Path = DEFAULT_WIND_CALIBRATION_PATH,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return target


def load_wind_calibrated_config(
    path: str | Path = DEFAULT_WIND_CALIBRATION_PATH,
) -> dict[str, Any] | None:
    source = Path(path)
    if not source.is_file():
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != WIND_CALIBRATION_SCHEMA:
        raise ValueError(f"unsupported Wind calibration schema: {source}")
    if payload.get("schema_version") != WIND_CALIBRATION_VERSION:
        raise ValueError(f"unsupported Wind calibration version: {source}")
    if not payload.get("met_target"):
        raise ValueError("refusing to use a Wind calibration that did not meet its target FAR")
    return payload
