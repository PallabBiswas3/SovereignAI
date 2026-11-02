"""Event-level false-alarm calibration for Wind SCADA normal-behavior residuals.

Why this exists
---------------
CARE is scored per *event*. A sample-level threshold does not directly control
whether a whole event alarms. Here every event is reduced to one scalar, and the alarm
threshold is the conformal quantile of that statistic over HEALTHY development
events, so the event-level false alarm rate is controlled directly:

    P(healthy event alarms) <= target_far        (marginal, finite-sample,
                                                  under exchangeability of
                                                  scorable healthy events,
                                                  with a fixed scoring rule)

Pipeline per event
------------------
1. Per-event robust scaling: median/MAD of the event's own healthy training
   residuals (per channel) -> z.
2. Fuse channels per sample (max |z| or RMS), then standardise the fused score
   with the training-partition median/MAD -> u.  (Use *held-out* training
   residuals if you can; in-sample residuals are optimistically small.  The
   development and test events must use the same procedure and have exchangeable
   resulting event scores.)
3. One scalar statistic over u:
     "persistence": max_t min(u[t..t+w-1])   (equivalently: the largest
                    threshold exceeded by a run of >= w consecutive samples)
     "max_smoothed": max_t median(u[t..t+w-1])
     "cusum": max one-sided CUSUM with drift `cusum_drift` (in MAD units)
4. alarm  <=>  statistic > conformal threshold.

Caveats
-------
* Statistics that take a maximum grow with event length.  Calibrate on healthy
  events whose prediction windows have a length comparable to the events you
  will score, or the FAR will not transfer.
* Only healthy events enter final calibration, so this controls false alarms,
  not recall. Select statistics on separate development tuning events; fit the
  final threshold on fresh healthy calibration units, never evaluation data.
* Wilson/bootstrap intervals are descriptive and assume independent validation
  events; they are not additional distribution-free guarantees for dependent
  events on a shared asset.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..uncertainty.conformal import conformal_quantile
from ..evaluation.split_manifest import DevelopmentSplitManifest
from .split_guard import LeakageError, SplitManifest, canonical_manifest

__all__ = [
    "EventResiduals",
    "EventScorer",
    "EventAlarmCalibration",
    "EventDecision",
    "EventFARCalibrator",
    "select_statistic",
    "wilson_interval",
    "early_warning_lead_time",
    "time_to_event_end",
    "CALIBRATION_CODE_VERSION",
]

CALIBRATION_CODE_VERSION = "event-far-1.1"
Statistic = Literal["persistence", "max_smoothed", "cusum"]


@dataclass(frozen=True)
class EventResiduals:
    """Residuals (observed - predicted) of one event.

    train:   residuals on the event's healthy training partition (n_t, c)
    predict: residuals on the partition to be scored (n_p, c)
    """

    train: np.ndarray
    predict: np.ndarray


def _as_2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a = a[:, None] if a.ndim == 1 else a
    if a.ndim != 2 or a.shape[1] < 1:
        raise ValueError("residuals must be a one- or two-dimensional array with channels")
    return np.where(np.isfinite(a), a, np.nan)


def _mad(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x, axis=0)
    return 1.4826 * np.nanmedian(np.abs(x - med), axis=0)


@dataclass(frozen=True)
class EventScorer:
    """Turns one event's residuals into a scalar statistic (and optional trace)."""

    statistic: Statistic = "persistence"
    window: int = 6
    fuse: Literal["max_abs", "rms"] = "max_abs"
    cusum_drift: float = 0.5
    max_missing_fraction: float = 0.5
    selection_manifest: Optional[DevelopmentSplitManifest] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.statistic not in {"persistence", "max_smoothed", "cusum"}:
            raise ValueError("unknown event statistic")
        if not isinstance(self.window, (int, np.integer)) or isinstance(self.window, bool) or self.window < 1:
            raise ValueError("window must be a positive integer")
        if self.fuse not in {"max_abs", "rms"}:
            raise ValueError("fuse must be max_abs or rms")
        if not np.isfinite(self.cusum_drift) or self.cusum_drift < 0:
            raise ValueError("cusum_drift must be finite and nonnegative")
        if not np.isfinite(self.max_missing_fraction) or not 0 <= self.max_missing_fraction <= 1:
            raise ValueError("max_missing_fraction must be in [0, 1]")

    # ------------------------------------------------------------------ #
    def _normalised_scores(self, ev: EventResiduals) -> Optional[np.ndarray]:
        tr, pr = _as_2d(ev.train), _as_2d(ev.predict)
        if tr.shape[1] != pr.shape[1]:
            raise ValueError("train/predict residuals must have the same channels")
        if len(tr) < 10 or not len(pr):
            return None
        # Avoid warning-driven all-NaN reductions; dead channels carry no evidence.
        usable = np.isfinite(tr).sum(axis=0) >= 10
        if not usable.any():
            return None
        tr, pr = tr[:, usable], pr[:, usable]
        med = np.nanmedian(tr, axis=0)
        mad = _mad(tr)
        std = np.nanstd(tr, axis=0)
        valid = (mad > 0) & (std > 0) & np.isfinite(mad)   # drop dead channels
        if not valid.any():
            return None

        def fused(x: np.ndarray) -> np.ndarray:
            z = np.abs((x[:, valid] - med[valid]) / mad[valid])
            if self.fuse == "max_abs":
                return np.nanmax(np.where(np.isnan(z), -np.inf, z), axis=1)
            count = np.isfinite(z).sum(axis=1)
            return np.sqrt(np.divide(np.nansum(z**2, axis=1), count,
                           out=np.full(len(z), np.nan), where=count > 0))

        s_tr = fused(tr)
        s_tr = s_tr[np.isfinite(s_tr)]
        if s_tr.size < 10:
            return None
        s_med = np.median(s_tr)
        s_mad = 1.4826 * np.median(np.abs(s_tr - s_med))
        if not s_mad > 0:
            return None
        s = fused(pr)
        missing = ~np.isfinite(s)
        if missing.mean() > self.max_missing_fraction:
            return None
        u = (s - s_med) / s_mad
        u[missing] = np.nan  # a missing row cannot complete a persistent alarm
        return u

    def trace(self, ev: EventResiduals) -> Optional[Tuple[np.ndarray, int]]:
        """Running statistic and the sample index at which trace[0] is available."""
        u = self._normalised_scores(ev)
        if u is None or u.size < self.window or not np.isfinite(u).any():
            return None
        w = self.window
        if self.statistic == "persistence":
            windows = sliding_window_view(u, w)
            series = np.where(np.isfinite(windows).all(axis=1), windows.min(axis=1), -np.inf)
            return (series, w - 1) if np.isfinite(series).any() else None
        if self.statistic == "max_smoothed":
            windows = sliding_window_view(u, w)
            series = np.where(np.isfinite(windows).all(axis=1), np.median(windows, axis=1), -np.inf)
            return (series, w - 1) if np.isfinite(series).any() else None
        if self.statistic == "cusum":
            out = np.empty_like(u)
            acc = 0.0
            for i, v in enumerate(u):
                if not np.isfinite(v):
                    acc, out[i] = 0.0, -np.inf
                    continue
                acc = max(0.0, acc + v - self.cusum_drift)
                out[i] = acc
            return out, 0
        raise ValueError(f"unknown statistic {self.statistic!r}")

    def score(self, ev: EventResiduals) -> float:
        """Scalar event statistic; NaN when the event cannot be scored."""
        t = self.trace(ev)
        return float("nan") if t is None else float(np.max(t[0]))


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if (not isinstance(k, (int, np.integer)) or not isinstance(n, (int, np.integer))
            or not 0 <= k <= n or not np.isfinite(z) or z <= 0):
        raise ValueError("need integer 0 <= k <= n and a finite positive z")
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def early_warning_lead_time(first_alarm: Optional[int], onset: int) -> Optional[int]:
    """Signed lead time vs fault onset, in samples: positive = alarm BEFORE onset."""
    return None if first_alarm is None else int(onset - first_alarm)


def time_to_event_end(first_alarm: Optional[int], event_end: int) -> Optional[int]:
    """Historical quantity `event_end - first_detection` (not an early-warning metric)."""
    return None if first_alarm is None else int(event_end - first_alarm)


# ---------------------------------------------------------------------- #
# Calibration
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class EventAlarmCalibration:
    """Frozen, serialisable calibration result."""

    statistic: str
    window: int
    fuse: str
    cusum_drift: float
    target_far: float
    threshold: float
    n_calibration: int
    n_excluded: int
    calibration_far: float
    n_validation: int
    validation_far: Optional[float]
    validation_far_wilson95: Optional[Tuple[float, float]]
    validation_far_bootstrap95: Optional[Tuple[float, float]]
    manifest_boundary: str
    manifest_sha256: str
    max_missing_fraction: float = 0.5
    n_validation_excluded: int = 0
    calibration_ids: Tuple[str, ...] = ()
    validation_ids: Tuple[str, ...] = ()
    selection_manifest_sha256: Optional[str] = None
    code_version: str = CALIBRATION_CODE_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2, allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> "EventAlarmCalibration":
        payload = json.loads(value)
        if not isinstance(payload, dict) or set(payload) != {item.name for item in fields(cls)}:
            raise ValueError("frozen calibration must contain the complete versioned configuration")
        if payload.get("code_version") != CALIBRATION_CODE_VERSION:
            raise ValueError("unsupported event calibration code version")
        for key in ("calibration_ids", "validation_ids", "validation_far_wilson95",
                    "validation_far_bootstrap95"):
            if payload.get(key) is not None:
                payload[key] = tuple(payload[key])
        result = cls(**payload)
        result.scorer()  # validate all score-affecting options
        if not np.isfinite(result.threshold) or not 0 < result.target_far < 1:
            raise ValueError("frozen calibration needs a finite threshold and valid target FAR")
        counts = (result.n_calibration, result.n_excluded, result.n_validation, result.n_validation_excluded)
        if any(type(count) is not int or count < 0 for count in counts) or result.n_calibration < 1:
            raise ValueError("frozen calibration needs valid sample counts")
        if (len(result.calibration_ids) != result.n_calibration + result.n_excluded
                or len(result.validation_ids) != result.n_validation + result.n_validation_excluded
                or len(set(result.calibration_ids)) != len(result.calibration_ids)
                or len(set(result.validation_ids)) != len(result.validation_ids)
                or set(result.calibration_ids) & set(result.validation_ids)):
            raise ValueError("frozen calibration unit IDs must match sample counts and remain disjoint")
        if len(result.manifest_sha256) != 64:
            raise ValueError("frozen calibration needs sample count and manifest checksum")
        try:
            bytes.fromhex(result.manifest_sha256)
        except ValueError as exc:
            raise ValueError("invalid manifest checksum") from exc
        result.to_json()  # refuse NaN/Infinity anywhere in a frozen artifact
        return result

    def scorer(self) -> EventScorer:
        return EventScorer(self.statistic, self.window, self.fuse, self.cusum_drift,
                           self.max_missing_fraction)

    def config_sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


@dataclass(frozen=True)
class EventDecision:
    alarm: Optional[bool]          # None = abstain
    status: str                    # "ok" | "abstain_unscorable"
    score: float
    threshold: float
    first_alarm_index: Optional[int] = None


class EventFARCalibrator:
    """Fit a target-event-FAR threshold on healthy development events."""

    def __init__(self, scorer: EventScorer, target_far: float = 0.10):
        if not 0.0 < target_far < 1.0:
            raise ValueError("target_far must be in (0, 1)")
        self.scorer = scorer
        self.target_far = target_far
        self.calibration_: Optional[EventAlarmCalibration] = None

    def fit(
        self,
        dev_events: Mapping[str, EventResiduals],
        manifest: SplitManifest | DevelopmentSplitManifest,
        validation_ids: Optional[Sequence[str]] = None,
        n_boot: int = 2000,
        seed: int = 0,
    ) -> EventAlarmCalibration:
        """dev_events must be HEALTHY and exactly the manifest's development ids.

        validation_ids: healthy development events held out from threshold
        fitting, used only to report an honest FAR estimate (use
        `group_split` so assets are not shared between calibration and
        validation when the boundary is asset/temporal).
        """
        # A failed refit must not silently leave a previous usable threshold.
        self.calibration_ = None
        manifest = canonical_manifest(manifest)
        try:
            manifest.assert_development_units(dev_events)
        except ValueError as exc:
            raise LeakageError(str(exc)) from exc
        if self.scorer.selection_manifest is not None:
            selection = canonical_manifest(self.scorer.selection_manifest)
            if selection.leakage_boundary != manifest.leakage_boundary:
                raise LeakageError("selection and final calibration must declare the same leakage boundary")
            # Protect both final calibration and evaluation from the tuning pool.
            canonical_manifest(replace(selection, evaluation_units=(
                manifest.development_units + manifest.evaluation_units)))
        if not isinstance(n_boot, (int, np.integer)) or n_boot < 0:
            raise ValueError("n_boot must be a nonnegative integer")
        val_ids = [] if validation_ids is None else list(validation_ids)
        if len(val_ids) != len(set(val_ids)):
            raise LeakageError("duplicate validation_ids")
        if not set(val_ids) <= set(dev_events):
            raise LeakageError("validation_ids must be a subset of the development events")
        cal_ids = [i for i in dev_events if i not in set(val_ids)]
        if val_ids:
            by_id = {unit.unit_id: unit for unit in manifest.development_units}
            canonical_manifest(replace(manifest,
                development_units=tuple(by_id[i] for i in cal_ids),
                evaluation_units=tuple(by_id[i] for i in val_ids)))

        cal_scores = np.array([self.scorer.score(dev_events[i]) for i in cal_ids])
        n_excluded = int((~np.isfinite(cal_scores)).sum())
        cal_scores = cal_scores[np.isfinite(cal_scores)]
        thr = conformal_quantile(cal_scores, self.target_far)
        if not np.isfinite(thr):
            need = int(np.ceil(1.0 / self.target_far - 1))
            raise ValueError(
                f"only {cal_scores.size} scorable healthy events; need at least {need} "
                f"for target_far={self.target_far}"
            )

        val_far = wilson = boot = None
        n_val = 0
        n_val_excluded = 0
        if val_ids:
            val_scores = np.array([self.scorer.score(dev_events[i]) for i in val_ids])
            n_val_excluded = int((~np.isfinite(val_scores)).sum())
            val_scores = val_scores[np.isfinite(val_scores)]
            n_val = int(val_scores.size)
            if n_val:
                k = int((val_scores > thr).sum())
                val_far = k / n_val
                wilson = wilson_interval(k, n_val)
                rng = np.random.default_rng(seed)
                fars = []
                for _ in range(n_boot):
                    t_b = conformal_quantile(
                        rng.choice(cal_scores, size=cal_scores.size, replace=True),
                        self.target_far,
                    )
                    held_out = rng.choice(val_scores, size=n_val, replace=True)
                    fars.append(float((held_out > t_b).mean()))
                if fars:
                    boot = (float(np.percentile(fars, 2.5)), float(np.percentile(fars, 97.5)))

        self.calibration_ = EventAlarmCalibration(
            statistic=self.scorer.statistic,
            window=int(self.scorer.window),
            fuse=self.scorer.fuse,
            cusum_drift=float(self.scorer.cusum_drift),
            target_far=float(self.target_far),
            threshold=float(thr),
            n_calibration=int(cal_scores.size),
            n_excluded=n_excluded,
            calibration_far=float((cal_scores > thr).mean()),
            n_validation=n_val,
            validation_far=val_far,
            validation_far_wilson95=wilson,
            validation_far_bootstrap95=boot,
            manifest_boundary=manifest.leakage_boundary,
            manifest_sha256=manifest.sha256(),
            max_missing_fraction=float(self.scorer.max_missing_fraction),
            n_validation_excluded=n_val_excluded,
            calibration_ids=tuple(sorted(cal_ids)), validation_ids=tuple(sorted(val_ids)),
            selection_manifest_sha256=(self.scorer.selection_manifest.sha256()
                                      if self.scorer.selection_manifest is not None else None),
        )
        return self.calibration_

    @classmethod
    def from_calibration(cls, calibration: EventAlarmCalibration) -> "EventFARCalibrator":
        frozen = EventAlarmCalibration.from_json(calibration.to_json())
        result = cls(frozen.scorer(), frozen.target_far)
        result.calibration_ = frozen
        return result

    def detect(self, ev: EventResiduals) -> EventDecision:
        """Apply the frozen threshold.  Unscorable events abstain, never alarm."""
        if self.calibration_ is None:
            raise RuntimeError("call fit() first")
        thr = self.calibration_.threshold
        # Detection follows the frozen configuration even if caller replaces .scorer.
        tr = self.calibration_.scorer().trace(ev)
        if tr is None:
            return EventDecision(None, "abstain_unscorable", float("nan"), thr)
        series, offset = tr
        score = float(series.max())
        first = None
        if score > thr:
            first = int(np.argmax(series > thr)) + offset
        return EventDecision(score > thr, "ok", score, thr, first)


def select_statistic(
    candidates: Sequence[EventScorer],
    dev_healthy: Mapping[str, EventResiduals],
    dev_anomalous: Mapping[str, EventResiduals],
    manifest: SplitManifest | DevelopmentSplitManifest,
    target_far: float = 0.10,
) -> Tuple[EventScorer, Dict[str, float]]:
    """Select on tuning events, then calibrate on a separate healthy pool.

    The manifest must declare exactly the union of healthy/anomalous tuning IDs.
    Provisional thresholds are for candidate selection only; they do not certify
    final FAR. The returned scorer carries the tuning manifest so final fit()
    rejects reused units/groups or invalid temporal ordering. Recall counts an
    unscorable anomaly as a miss, rather than improving recall by excluding it.
    """
    manifest = canonical_manifest(manifest)
    if set(dev_healthy) & set(dev_anomalous):
        raise LeakageError("healthy and anomalous tuning IDs must be disjoint")
    if not dev_anomalous:
        raise ValueError("statistic selection needs development anomalies")
    try:
        manifest.assert_development_units((*dev_healthy, *dev_anomalous))
    except ValueError as exc:
        raise LeakageError(str(exc)) from exc
    healthy_manifest = replace(manifest, development_units=tuple(
        unit for unit in manifest.development_units if unit.unit_id in dev_healthy))
    best, best_recall, report = None, -1.0, {}
    for sc in candidates:
        cal = EventFARCalibrator(sc, target_far)
        try:
            cal.fit(dev_healthy, healthy_manifest, n_boot=0)
        except LeakageError:
            raise
        except ValueError:
            continue
        dets = [cal.detect(ev) for ev in dev_anomalous.values()]
        recall = float(np.mean([d.alarm is True for d in dets]))
        key = f"{sc.statistic}/w{sc.window}/{sc.fuse}/drift{sc.cusum_drift}/missing{sc.max_missing_fraction}"
        report[key] = recall
        if recall > best_recall:
            best, best_recall = sc, recall
    if best is None:
        raise ValueError("no candidate produced a usable threshold")
    return replace(best, selection_manifest=manifest), report
