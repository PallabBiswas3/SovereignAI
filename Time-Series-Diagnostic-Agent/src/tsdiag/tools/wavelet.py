from __future__ import annotations

import numpy as np
import pywt
from scipy.signal import correlate, correlation_lags


def _as_2d(x):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("expected [samples, channels]")
    return arr


def wavelet_denoising(signal_matrix, wavelet: str = "db4", level: int | None = None, threshold_rule: str = "soft"):
    x = _as_2d(signal_matrix)
    out = np.empty_like(x)
    metadata = []
    for j in range(x.shape[1]):
        col = x[:, j]
        max_level = pywt.dwt_max_level(len(col), pywt.Wavelet(wavelet).dec_len)
        use_level = min(level if level is not None else max(1, min(5, max_level)), max_level)
        coeffs = pywt.wavedec(col, wavelet, level=use_level)
        detail = coeffs[-1]
        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745 if detail.size else 0.0
        threshold = sigma * np.sqrt(2 * np.log(max(len(col), 2)))
        clean = [coeffs[0]] + [pywt.threshold(c, threshold, mode=threshold_rule) for c in coeffs[1:]]
        recon = pywt.waverec(clean, wavelet)[:len(col)]
        out[:, j] = recon
        metadata.append({"channel": j, "wavelet": wavelet, "level": use_level, "threshold": float(threshold), "rule": threshold_rule})
    return {"denoised_signal_matrix": out, "denoising_metadata": metadata}


def cross_correlation_analysis(denoised_signal_matrix, max_lag: int | None = None):
    x = _as_2d(denoised_signal_matrix)
    n, d = x.shape
    pairwise = []
    energy = np.zeros(d, dtype=float)
    for i in range(d):
        for j in range(i + 1, d):
            a = x[:, i] - np.mean(x[:, i])
            b = x[:, j] - np.mean(x[:, j])
            corr = correlate(a, b, mode="full", method="fft")
            lags = correlation_lags(len(a), len(b), mode="full")
            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
            corr = corr / denom
            if max_lag is not None:
                mask = np.abs(lags) <= max_lag
                corr, lags = corr[mask], lags[mask]
            k = int(np.argmax(np.abs(corr)))
            peak = float(corr[k])
            lag = int(lags[k])
            pairwise.append({"pair": [i, j], "peak_correlation": peak, "lag_samples": lag})
            e = float(np.sum(corr**2))
            energy[i] += e
            energy[j] += e
    return {"pairwise_correlations": pairwise, "correlation_energy": energy}


def correlation_sensor_weighting(correlation_energy):
    e = np.asarray(correlation_energy, dtype=float)
    e = np.clip(e, 0, None)
    if e.sum() <= 1e-12:
        w = np.ones_like(e) / max(len(e), 1)
    else:
        w = e / e.sum()
    return {"sensor_weights": w}


def multisensor_fusion(denoised_signal_matrix, sensor_weights):
    x = _as_2d(denoised_signal_matrix)
    w = np.asarray(sensor_weights, dtype=float).ravel()
    if w.size != x.shape[1]:
        raise ValueError("sensor_weights length must match number of channels")
    s = w.sum()
    if s <= 0:
        raise ValueError("sensor_weights must have positive total weight")
    w = w / s
    return {"fused_waveform": x @ w}
