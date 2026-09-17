from __future__ import annotations

import numpy as np


def _as_2d(x):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("expected [samples, channels]")
    return arr


def rolling_statistics(residuals, timestamps=None, window: int = 20):
    x = _as_2d(residuals)
    if window < 3:
        raise ValueError("window must be >= 3")
    n, d = x.shape
    means = np.full((n, d), np.nan)
    stds = np.full((n, d), np.nan)
    slopes = np.full((n, d), np.nan)
    for i in range(window - 1, n):
        seg = x[i - window + 1:i + 1]
        t = np.arange(window, dtype=float)
        means[i] = np.nanmean(seg, axis=0)
        stds[i] = np.nanstd(seg, axis=0)
        for j in range(d):
            col = seg[:, j]
            finite = np.isfinite(col)
            if finite.sum() >= 3:
                slopes[i, j] = np.polyfit(t[finite], col[finite], 1)[0]
    return {"rolling_mean": means, "rolling_std": stds, "rolling_slope": slopes}


def cross_sensor_relationships(signal_matrix, channel_names=None, relationship_pairs=None, reference_correlations=None):
    x = _as_2d(signal_matrix)
    names = channel_names or [f"ch{i}" for i in range(x.shape[1])]
    corr = np.corrcoef(x, rowvar=False)
    pairs = relationship_pairs
    if pairs is None:
        pairs = [(i, j) for i in range(x.shape[1]) for j in range(i + 1, x.shape[1])]
    relationships = []
    shifts = []
    for a, b in pairs:
        ia = names.index(a) if isinstance(a, str) else int(a)
        ib = names.index(b) if isinstance(b, str) else int(b)
        value = float(corr[ia, ib])
        item = {"pair": [names[ia], names[ib]], "correlation": value}
        if reference_correlations is not None:
            ref = np.asarray(reference_correlations, dtype=float)
            shift = float(abs(value - ref[ia, ib]))
            item["shift"] = shift
            shifts.append(item)
        relationships.append(item)
    return {"correlations": relationships, "relationship_shift_scores": shifts}


def _robust_scale(seg: np.ndarray) -> np.ndarray:
    center = np.nanmedian(seg, axis=0)
    mad = np.nanmedian(np.abs(seg - center), axis=0) * 1.4826
    fallback = np.nanstd(seg, axis=0)
    return np.where(mad > 1e-9, mad, np.where(fallback > 1e-9, fallback, 1e-9))


def change_point_detection(residuals, timestamps=None, min_size: int = 20, z_threshold: float = 4.0, regime_ids=None, variance_model=None):
    """Deterministic robust mean-shift detector using left/right median contrasts.

    The contrast is normalized by *within-window* robust noise rather than a global
    scale. A global MAD can become inflated by the very regime shift we are trying
    to detect, which suppresses clear step changes. This remains a simple baseline
    for later PELT/BOCPD comparisons.
    """
    x = _as_2d(residuals)
    n, d = x.shape
    if n < 2 * min_size + 1:
        return {"change_points": [], "change_magnitudes": []}

    candidates = []
    for i in range(min_size, n - min_size):
        left_seg = x[i - min_size:i]
        right_seg = x[i:i + min_size]
        left = np.nanmedian(left_seg, axis=0)
        right = np.nanmedian(right_seg, axis=0)
        left_scale = _robust_scale(left_seg)
        right_scale = _robust_scale(right_seg)
        pooled = np.sqrt(0.5 * (left_scale**2 + right_scale**2)) + 1e-12
        contrast = np.abs(right - left)
        score = float(np.nanmax(contrast / pooled))
        if score >= z_threshold:
            candidates.append((i, score, contrast))

    selected = []
    for idx, score, mag in sorted(candidates, key=lambda z: z[1], reverse=True):
        if all(abs(idx - kept[0]) >= min_size for kept in selected):
            selected.append((idx, score, mag))
    selected.sort(key=lambda z: z[0])

    cps = []
    mags = []
    for idx, score, mag in selected:
        time_value = timestamps[idx] if timestamps is not None else idx
        cps.append({"index": int(idx), "time": time_value, "score": score})
        mags.append({"index": int(idx), "per_channel": np.asarray(mag).tolist(), "max_magnitude": float(np.max(mag))})
    return {"change_points": cps, "change_magnitudes": mags}
