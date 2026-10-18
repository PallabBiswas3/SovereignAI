"""Additive, causal mean-shift alerts; never certify a physical fault or root.

Reference rolling-mean variance reflects serial dependence without learning
from current observations. Multiple horizons capture abrupt and gradual shifts.
The three-sigma rule is a sensitivity heuristic, not a calibrated probability
or a promise that every fault is detectable.
"""
import numpy as np

from ..tools.monitoring import apply_alarm_persistence


def _rolling_mean(x, window):
    total = np.vstack((np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)))
    return (total[window:] - total[:-window]) / window


def sustained_process_alert(signal, reference):
    x, ref = np.asarray(signal, dtype=float), np.asarray(reference, dtype=float)
    if (x.ndim != 2 or ref.ndim != 2 or not len(x) or len(ref) < 8
            or x.shape[1] != ref.shape[1] or not x.shape[1]
            or not np.isfinite(x).all() or not np.isfinite(ref).all()):
        raise ValueError('finite nonempty aligned process matrices and healthy reference required')
    center = ref.mean(axis=0)
    point_scale = ref.std(axis=0)
    score = np.zeros(len(x))
    horizons = []
    for window in (8, 32, 128):
        if len(ref) < 2 * window or len(x) < window:
            continue
        reference_means = _rolling_mean(ref, window)
        # Floor protects against degenerate references, but does not certify them.
        scale = np.maximum(reference_means.std(axis=0), point_scale / np.sqrt(window))
        scale = np.maximum(scale, 1e-12)
        means = _rolling_mean(x, window)
        channel_score = np.abs(means - center) / scale
        score[window - 1:] = np.maximum(score[window - 1:], channel_score.max(axis=1))
        horizons.append(window)
    mask = apply_alarm_persistence(score >= 3., 3)
    indices = np.flatnonzero(mask)
    return {'detected': bool(len(indices)), 'alarm_mask': mask, 'score': score,
            'first_alert_index': int(indices[0]) if len(indices) else None,
            'method': 'reference_multiscale_mean_v1', 'horizons': horizons,
            'threshold': 3., 'min_consecutive': 3,
            'interpretation': 'possible_deviation_not_confirmed_fault',
            'calibrated_probability': False}
