from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# Canonical Wind-SCADA CUSUM configuration. Keep the runner, plugin, benchmark
# API, CLI and CI on these values unless a benchmark explicitly declares a
# different protocol.
DEFAULT_CUSUM_DRIFT = 0.5
DEFAULT_CUSUM_THRESHOLD = 10.0
DEFAULT_CUSUM_HOLD_SAMPLES = 6


@dataclass(frozen=True)
class ResidualCUSUMConfig:
    # k=0.5 is the standard half-sigma reference value for detecting an
    # approximately one-sigma persistent mean shift. The previous 0.25 value
    # was too eager when several SCADA residual channels were monitored in
    # parallel.
    drift: float = DEFAULT_CUSUM_DRIFT
    threshold: float = DEFAULT_CUSUM_THRESHOLD
    reset_after_alarm: bool = True
    hold_samples: int = DEFAULT_CUSUM_HOLD_SAMPLES
    two_sided: bool = True
    # Aggregate multichannel alarms should require corroboration when requested,
    # while preserving every per-channel crossing for localization/evidence.
    # Generic/univariate CUSUM keeps the backwards-compatible value of 1.
    min_channel_support: int = 1


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("expected [samples, channels]")
    return arr


def residual_cusum(
    normalized_residuals,
    *,
    timestamps=None,
    reset_mask=None,
    config: ResidualCUSUMConfig | None = None,
) -> dict[str, Any]:
    """Sequential two-sided CUSUM for low-amplitude persistent residual shifts.

    Inputs are normalized healthy-model residuals. ``reset_mask`` may mark known
    operating-boundary samples (for example, a regime transition). Accumulators
    are cleared at those boundaries so a normal change in operating point cannot
    be integrated into a false persistent fault.

    Per-channel CUSUM crossings are always retained. ``min_channel_support``
    controls only the aggregate alarm stream, preventing the family-wise false
    alarm inflation caused by taking a simple union over many monitored channels.
    Corroboration is evaluated over each channel's short hold window rather than
    requiring multiple channels to cross on the exact same sample. Hold windows
    are truncated at operating-boundary resets so evidence cannot leak across
    regimes.
    """
    cfg = config or ResidualCUSUMConfig()
    x = _as_2d(normalized_residuals)
    x = np.where(np.isfinite(x), x, 0.0)
    n, d = x.shape

    min_support = max(1, int(cfg.min_channel_support))
    if d:
        min_support = min(min_support, d)

    ts = None
    if timestamps is not None:
        ts = np.asarray(timestamps)
        if ts.shape[0] != n:
            raise ValueError("timestamps length mismatch")

    resets = np.zeros(n, dtype=bool)
    if reset_mask is not None:
        resets = np.asarray(reset_mask, dtype=bool)
        if resets.shape != (n,):
            raise ValueError("reset_mask length mismatch")

    pos = np.zeros(d, dtype=float)
    neg = np.zeros(d, dtype=float)
    pos_history = np.zeros((n, d), dtype=float)
    neg_history = np.zeros((n, d), dtype=float)
    point_mask = np.zeros((n, d), dtype=bool)

    for i in range(n):
        if resets[i]:
            pos[:] = 0.0
            neg[:] = 0.0
        row = x[i]
        pos = np.maximum(0.0, pos + row - float(cfg.drift))
        if cfg.two_sided:
            neg = np.maximum(0.0, neg - row - float(cfg.drift))
        else:
            neg[:] = 0.0
        triggered = (pos >= float(cfg.threshold)) | (neg >= float(cfg.threshold))
        point_mask[i] = triggered
        pos_history[i] = pos
        neg_history[i] = neg
        if cfg.reset_after_alarm and np.any(triggered):
            pos[triggered] = 0.0
            neg[triggered] = 0.0

    hold_samples = max(1, int(cfg.hold_samples))
    channel_hold_mask = np.zeros((n, d), dtype=bool)
    channel_hold_score = np.zeros((n, d), dtype=float)
    strength = np.maximum(pos_history, neg_history)

    for channel in range(d):
        for idx in np.flatnonzero(point_mask[:, channel]):
            stop = min(n, idx + hold_samples)
            # A pre-transition crossing must not corroborate a crossing from the
            # next operating regime. Truncate its evidence window at the first
            # reset strictly after the crossing.
            future_resets = np.flatnonzero(resets[idx + 1:stop])
            if future_resets.size:
                stop = idx + 1 + int(future_resets[0])
            if stop <= idx:
                continue
            channel_hold_mask[idx:stop, channel] = True
            trigger_strength = float(strength[idx, channel])
            channel_hold_score[idx:stop, channel] = np.maximum(
                channel_hold_score[idx:stop, channel], trigger_strength
            )

    channel_support_count = np.sum(channel_hold_mask, axis=1).astype(int)
    channel_support_fraction = channel_support_count / max(d, 1)
    aggregate_active = channel_support_count >= min_support

    # A change point is the onset of a corroborated episode, not every sample
    # inside the episode's evidence window.
    aggregate_onsets = aggregate_active.copy()
    if n > 1:
        aggregate_onsets[1:] &= ~aggregate_active[:-1]

    channel_max_score = np.nanmax(strength, axis=1) if d else np.zeros(n)
    if d:
        sorted_held_scores = np.sort(channel_hold_score, axis=1)
        aggregate_score = sorted_held_scores[:, -min_support]
    else:
        aggregate_score = np.zeros(n)

    change_points = np.flatnonzero(aggregate_onsets).astype(int).tolist()
    channel_change_points = {
        int(j): np.flatnonzero(point_mask[:, j]).astype(int).tolist()
        for j in range(d)
        if np.any(point_mask[:, j])
    }

    def _serialize_timestamp(value):
        if isinstance(value, np.datetime64):
            return str(value)
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    change_point_timestamps = [] if ts is None else [_serialize_timestamp(ts[i]) for i in change_points]
    channel_change_point_timestamps = {}
    if ts is not None:
        channel_change_point_timestamps = {
            int(channel): [_serialize_timestamp(ts[i]) for i in indices]
            for channel, indices in channel_change_points.items()
        }

    return {
        "change_points": change_points,
        "change_point_timestamps": change_point_timestamps,
        "channel_change_points": channel_change_points,
        "channel_change_point_timestamps": channel_change_point_timestamps,
        "change_point_mask": aggregate_onsets,
        "aggregate_support_mask": aggregate_active,
        "channel_change_point_mask": point_mask,
        "channel_hold_mask": channel_hold_mask,
        "channel_support_count": channel_support_count,
        "channel_support_fraction": channel_support_fraction,
        "alarm_mask": aggregate_active,
        "cusum_scores": aggregate_score,
        "channel_max_cusum_scores": channel_max_score,
        "positive_cusum": pos_history,
        "negative_cusum": neg_history,
        "reset_mask": resets,
        "configuration": {
            "drift": float(cfg.drift),
            "threshold": float(cfg.threshold),
            "reset_after_alarm": bool(cfg.reset_after_alarm),
            "hold_samples": hold_samples,
            "two_sided": bool(cfg.two_sided),
            "min_channel_support": min_support,
        },
    }
