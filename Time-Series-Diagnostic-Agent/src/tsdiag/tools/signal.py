from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, hilbert, stft, welch
from scipy.stats import skew


def _as_1d(signal) -> np.ndarray:
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 16:
        raise ValueError("signal must contain at least 16 samples")
    return x


def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def signal_integrity(signal, sampling_rate_hz: float, clipping_quantile: float = 0.999):
    x = np.asarray(signal, dtype=float).ravel()
    flags = []
    if x.size < 16:
        flags.append("too_short")
    nonfinite = ~np.isfinite(x)
    if np.any(nonfinite):
        flags.append("nan_or_inf")
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {"quality_flags": flags or ["no_finite_samples"], "duration_s": 0.0, "clipping_fraction": 1.0, "dc_offset": np.nan}
    if sampling_rate_hz <= 0:
        flags.append("invalid_sampling_rate")

    lo, hi = np.quantile(finite, [1 - clipping_quantile, clipping_quantile])
    span = hi - lo
    tol = max(abs(span) * 1e-6, 1e-12)
    at_extreme = (np.abs(finite - finite.min()) <= tol) | (np.abs(finite - finite.max()) <= tol)
    clipping = float(np.mean(at_extreme))
    plateau_run = _longest_true_run(at_extreme)
    # Repeated isolated extrema are normal for periodic signals. ADC clipping usually
    # creates a flat plateau over consecutive samples, so require both occupancy and
    # a persistent run before flagging.
    if clipping > 0.01 and plateau_run >= 3:
        flags.append("possible_clipping")

    dc = float(np.mean(finite))
    rms = float(np.sqrt(np.mean((finite - dc) ** 2)))
    if rms > 0 and abs(dc) > 0.5 * rms:
        flags.append("large_dc_offset")
    return {
        "quality_flags": flags,
        "duration_s": float(finite.size / sampling_rate_hz) if sampling_rate_hz > 0 else 0.0,
        "clipping_fraction": clipping,
        "clipping_plateau_run": int(plateau_run),
        "dc_offset": dc,
        "finite_fraction": float(finite.size / max(x.size, 1)),
    }


def time_domain_features(signal):
    x = _as_1d(signal)
    x = x[np.isfinite(x)]
    centered = x - x.mean()
    rms = float(np.sqrt(np.mean(centered**2)))
    peak = float(np.max(np.abs(centered)))
    m2 = float(np.mean(centered**2))
    m4 = float(np.mean(centered**4))
    return {
        "rms": rms,
        "std": float(np.std(x)),
        "peak": peak,
        "crest_factor": float(peak / rms) if rms > 0 else 0.0,
        "skewness": float(skew(x, bias=False)) if x.size > 2 else 0.0,
        "kurtosis": float(m4 / (m2**2)) if m2 > 0 else 0.0,
    }


def welch_psd(signal, sampling_rate_hz: float, nperseg: int | None = None, top_k: int = 8):
    x = _as_1d(signal)
    nperseg = min(nperseg or 2048, x.size)
    f, p = welch(x - x.mean(), fs=sampling_rate_hz, nperseg=nperseg)
    peaks, props = find_peaks(p)
    if peaks.size:
        order = peaks[np.argsort(p[peaks])[::-1][:top_k]]
        dominant = [{"frequency_hz": float(f[i]), "power": float(p[i])} for i in order]
    else:
        dominant = []
    return {"frequency_hz": f, "psd": p, "dominant_peaks": dominant}


def time_frequency_analysis(signal, sampling_rate_hz: float, nperseg: int = 512, noverlap: int | None = None):
    x = _as_1d(signal)
    nperseg = min(nperseg, x.size)
    if noverlap is None:
        noverlap = int(0.75 * nperseg)
    noverlap = min(noverlap, nperseg - 1)
    f, t, z = stft(x - x.mean(), fs=sampling_rate_hz, nperseg=nperseg, noverlap=noverlap, boundary=None)
    power = np.abs(z) ** 2
    baseline = np.median(power, axis=1, keepdims=True) + 1e-12
    ratio = power / baseline
    transient_mask = ratio > 6.0
    transient_regions = []
    for fi, ti in np.argwhere(transient_mask)[:500]:
        transient_regions.append({"frequency_hz": float(f[fi]), "time_s": float(t[ti]), "ratio": float(ratio[fi, ti])})
    return {"time_axis": t, "frequency_axis": f, "time_frequency_map": power, "transient_regions": transient_regions}


def spectral_kurtosis(signal, sampling_rate_hz: float, nperseg: int = 512, noverlap: int | None = None):
    tf = time_frequency_analysis(signal, sampling_rate_hz, nperseg=nperseg, noverlap=noverlap)
    power = tf["time_frequency_map"]
    f = tf["frequency_axis"]
    centered = power - power.mean(axis=1, keepdims=True)
    m2 = np.mean(centered**2, axis=1)
    m4 = np.mean(centered**4, axis=1)
    scores = np.divide(m4, m2**2, out=np.zeros_like(m4), where=m2 > 1e-20)
    valid = (f > max(10.0, 0.01 * sampling_rate_hz)) & (f < 0.48 * sampling_rate_hz)
    if not np.any(valid):
        return {"band_scores": [], "recommended_band_hz": None}
    idx = np.flatnonzero(valid)[np.argmax(scores[valid])]
    df = float(f[1] - f[0]) if f.size > 1 else sampling_rate_hz / 2
    half_width = max(4 * df, 0.04 * sampling_rate_hz)
    band = [max(1.0, float(f[idx] - half_width)), min(float(sampling_rate_hz / 2 * 0.98), float(f[idx] + half_width))]
    top = np.argsort(scores[valid])[::-1][:10]
    valid_idx = np.flatnonzero(valid)[top]
    band_scores = [{"frequency_hz": float(f[i]), "kurtosis": float(scores[i])} for i in valid_idx]
    return {"band_scores": band_scores, "recommended_band_hz": band}


def bandpass_filter(signal, sampling_rate_hz: float, band_hz, order: int = 4):
    x = _as_1d(signal)
    low, high = map(float, band_hz)
    nyq = sampling_rate_hz / 2
    if not (0 < low < high < nyq):
        raise ValueError("band must satisfy 0 < low < high < Nyquist")
    b, a = butter(order, [low, high], btype="bandpass", fs=sampling_rate_hz)
    y = filtfilt(b, a, x)
    return {"filtered_signal": y, "filter_metadata": {"type": "butterworth", "order": order, "band_hz": [low, high]}}


def hilbert_envelope(filtered_signal):
    x = _as_1d(filtered_signal)
    return {"envelope": np.abs(hilbert(x))}


def envelope_spectrum(envelope, sampling_rate_hz: float, top_k: int = 12):
    x = _as_1d(envelope)
    x = x - x.mean()
    window = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * window)) ** 2
    freq = np.fft.rfftfreq(x.size, 1 / sampling_rate_hz)
    peaks, _ = find_peaks(spec)
    order = peaks[np.argsort(spec[peaks])[::-1][:top_k]] if peaks.size else np.array([], dtype=int)
    return {
        "frequency_hz": freq,
        "envelope_power": spec,
        "peaks": [{"frequency_hz": float(freq[i]), "power": float(spec[i])} for i in order],
    }
