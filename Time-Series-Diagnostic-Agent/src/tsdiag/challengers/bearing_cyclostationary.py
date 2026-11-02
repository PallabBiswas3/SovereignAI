"""Cyclostationary bearing features: modulation spectrum, improved envelope
spectrum (IES), order-domain envelope, and fault-frequency scores.

Why
---
A classical envelope spectrum picks one band and one demodulation.  Bearing
faults are *cyclostationary*: the squared envelope of the resonance bands is
modulated at the fault repetition rate.  Computing that modulation in every
frequency band (the cyclic modulation spectrum) and combining the bands by how
impulsive they are (spectral-kurtosis weights) gives a more robust fault
signature when the resonance band is unknown or tonal interference (gear mesh)
is present. This STFT-power modulation proxy is not a full cyclic-coherence
estimator, and improved discrimination must be established on independent data.

Pipeline
--------
1. STFT power P(f, m).  Gaussian noise gives E[P^2]/E[P]^2 = 2, so
   SK(f) = E[P^2]/E[P]^2 - 2 is ~0 for noise and >0 for impulsive bands.
2. Cyclic modulation depth  gamma(f, alpha) = |FFT_m(P(f, .) - mean)| / mean,
   the relative modulation of band f at cyclic frequency alpha.
3. IES(alpha) = sum_f w_f gamma(f, alpha) with w_f proportional to max(SK(f), 0).
4. Fault score per characteristic frequency: peak (within +/- slip tolerance) over
   a robust local noise floor, log-averaged over harmonics.
5. Varying speed: demodulate with a band-pass + Hilbert envelope, resample the
   envelope to the shaft angle, and read fault ORDERS (fault frequency / shaft
   frequency).

Parameters and limits
---------------------
* The STFT frame (`nperseg`/fs) must be SHORTER than the fault period 1/f_fault,
  otherwise individual impulses are not resolved and the modulation washes out.
  The defaults (64 samples, hop 8) suit fs of about 12 kHz and faults of
  roughly 50-200 Hz; scale them with fs and the lowest fault frequency.
* Sensitivity depends on record length, speed, bandwidth and noise. No universal
  SNR cutoff or fault-detection guarantee follows from the synthetic examples.
  An unscorable NaN must lead the caller to abstain; a finite score below the
  healthy threshold is only a lack of detection evidence.

Thresholds
----------
Do not hand-set a detection threshold.  Compute `max_fault_score` on healthy
development recordings and take `healthy_threshold(...)`, a conformal quantile
that bounds marginal healthy false alarms only for a fixed score and exchangeable
calibration/test units. Split by independent recording/specimen. This is not a
fault-type classifier or a per-record probability of failure.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, stft

from ..uncertainty.conformal import conformal_quantile

__all__ = [
    "stft_power",
    "spectral_kurtosis_bands",
    "select_resonance_band",
    "cyclic_modulation_depth",
    "improved_envelope_spectrum",
    "fault_score",
    "bearing_cyclo_features",
    "max_fault_score",
    "healthy_threshold",
    "angular_resample",
    "order_envelope_spectrum",
    "order_fault_scores",
]

_EPS = 1e-12


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _integer(value, name, minimum=1):
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _signal(x, minimum=2):
    x = np.asarray(x, float)
    if x.ndim != 1 or len(x) < minimum or not np.isfinite(x).all():
        raise ValueError("signal must be a sufficiently long finite one-dimensional array")
    return x


def _power(P):
    P = np.asarray(P, float)
    if (P.ndim != 2 or P.shape[0] < 3 or P.shape[1] < 4
            or not np.isfinite(P).all() or np.any(P < 0)):
        raise ValueError("power must be a finite nonnegative (F>=3, M>=4) matrix")
    return P


def _band(band, fs, *, strict=False):
    if len(band) != 2 or not np.isfinite(band).all():
        raise ValueError("band must contain two finite frequencies")
    lo, hi = map(float, band)
    if not 0 <= lo < hi <= fs / 2 or (strict and (lo == 0 or hi == fs / 2)):
        raise ValueError("band must lie inside the sampling Nyquist range")
    return lo, hi


# --------------------------------------------------------------------------- #
# Time-frequency building blocks
# --------------------------------------------------------------------------- #
def stft_power(x, fs: float, nperseg: int = 64, hop: int = 8):
    """STFT power P (F, M), frequency axis, and the frame rate fs/hop."""
    _positive(fs, "fs")
    _integer(nperseg, "nperseg", 8)
    _integer(hop, "hop")
    if nperseg % 2 or hop > nperseg:
        raise ValueError("use an even nperseg and 1 <= hop <= nperseg")
    x = _signal(x, nperseg + 3 * hop)
    f, _, Z = stft(x - x.mean(), fs, window="hann", nperseg=nperseg,
                   noverlap=nperseg - hop, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    return f, fs / hop, _power(P)


def spectral_kurtosis_bands(P: np.ndarray) -> np.ndarray:
    """SK(f) = E[P^2]/E[P]^2 - 2 per frequency bin (about 0 for Gaussian noise).

    The DC and Nyquist bins of a real-valued STFT have one degree of freedom
    (noise gives E[P^2]/E[P]^2 = 3, i.e. SK = 1), so they are set to 0 or they
    would dominate the band weighting.
    """
    P = _power(P)
    m1 = P.mean(axis=1)
    relative = np.divide(P, m1[:, None], out=np.zeros_like(P), where=m1[:, None] > 0)
    sk = np.where(m1 > 0, (relative**2).mean(axis=1) - 2.0, 0.0)
    sk[0] = 0.0
    sk[-1] = 0.0
    return sk


def select_resonance_band(x, fs: float, bandwidth_hz: float = 1000.0, nperseg: int = 64,
                          hop: int = 8, *, energy_weighted: bool = True) -> Tuple[float, float]:
    """Select an interior band using positive SK, optionally weighted by energy."""
    _positive(bandwidth_hz, "bandwidth_hz")
    f, _, P = stft_power(x, fs, nperseg, hop)
    sk = np.maximum(spectral_kurtosis_bands(P), 0)
    energy = P.mean(axis=1)
    if not energy.max() > 0:
        raise ValueError("a silent/constant signal has no scorable resonance band")
    if energy_weighted:
        sk *= energy / energy.max()
    f, sk = f[1:-1], sk[1:-1]
    if not sk.max() > 0:
        raise ValueError("no impulsive resonance band is supported by the supplied recording")
    df = f[1] - f[0]
    w = max(int(round(bandwidth_hz / df)), 1)
    if w > len(f):
        raise ValueError("bandwidth exceeds the resolvable interior frequency range")
    cs = np.concatenate([[0.0], np.cumsum(sk)])
    means = (cs[w:] - cs[:-w]) / w
    i = int(np.argmax(means))
    return float(f[i] - df / 2), float(f[i + w - 1] + df / 2)


def cyclic_modulation_depth(P: np.ndarray, frame_rate: float):
    """gamma(f, alpha): relative modulation depth of each band at each cyclic frequency.

    Returns (alpha, gamma) with gamma of shape (F, n_alpha).
    """
    P = _power(P)
    _positive(frame_rate, "frame_rate")
    M = P.shape[1]
    mean = P.mean(axis=1, keepdims=True)
    win = np.hanning(M)
    spec = np.abs(np.fft.rfft((P - mean) * win[None, :], axis=1))
    spec = spec * 2.0 / win.sum()
    gamma = np.divide(spec, mean, out=np.zeros_like(spec), where=mean > 0)
    return np.fft.rfftfreq(M, 1.0 / frame_rate), gamma


def improved_envelope_spectrum(
    x,
    fs: float,
    band: Optional[Tuple[float, float]] = None,
    nperseg: int = 64,
    hop: int = 8,
):
    """Kurtosis-weighted average of the cyclic modulation depth over bands.

    band restricts the frequency bins used (e.g. an expected resonance region).
    Returns (alpha_hz, ies).
    """
    f, frate, P = stft_power(x, fs, nperseg, hop)
    alpha, gamma = cyclic_modulation_depth(P, frate)
    sk = np.clip(spectral_kurtosis_bands(P), 0.0, None)
    energy = P.mean(axis=1)
    sk *= energy / energy.max() if energy.max() > 0 else np.zeros_like(energy)
    if band is not None:
        band = _band(band, fs)
        if not ((f >= band[0]) & (f <= band[1])).any():
            raise ValueError("band contains no resolved STFT frequency bins")
        sk = np.where((f >= band[0]) & (f <= band[1]), sk, 0.0)
    if sk.sum() <= _EPS:
        sk = energy.copy()
        if band is not None:
            sk = np.where((f >= band[0]) & (f <= band[1]), sk, 0)
    if not sk.sum() > 0:
        return alpha, np.zeros_like(alpha)
    w = sk / sk.sum()
    return alpha, (w[:, None] * gamma).sum(axis=0)


# --------------------------------------------------------------------------- #
# Fault-frequency scoring
# --------------------------------------------------------------------------- #
def fault_score(
    axis: np.ndarray,
    spectrum: np.ndarray,
    f0: float,
    harmonics: Sequence[int] = (1, 2, 3),
    tol: float = 0.03,
    floor_halfwidth: float = 0.25,
) -> float:
    """Mean log(peak / local noise floor) over harmonics of f0.

    tol is the relative slip tolerance for the peak search; the noise floor is
    the median of the spectrum within +/- floor_halfwidth*f0 of each harmonic,
    excluding the peak window.  Harmonics above the axis range are skipped.
    """
    axis, spectrum = np.asarray(axis, float), np.asarray(spectrum, float)
    if (axis.ndim != 1 or len(axis) < 3 or axis.shape != spectrum.shape
            or not np.isfinite(axis).all() or not np.isfinite(spectrum).all()
            or np.any(axis < 0) or np.any(np.diff(axis) <= 0) or np.any(spectrum < 0)):
        raise ValueError("spectrum must be finite/nonnegative and aligned with increasing frequencies")
    _positive(f0, "fault frequency/order")
    _positive(floor_halfwidth, "floor_halfwidth")
    if not np.isfinite(tol) or not 0 <= tol < 1:
        raise ValueError("tol must be in [0, 1)")
    harmonics = tuple(harmonics)
    if not harmonics or len(set(harmonics)) != len(harmonics):
        raise ValueError("harmonics must be nonempty and unique")
    for harmonic in harmonics:
        _integer(harmonic, "harmonic")
    if not spectrum.max() > 0:
        return float("nan")
    scores = []
    for h in harmonics:
        fc = h * f0
        if fc * (1 + tol) >= axis[-1]:
            continue
        peak_mask = (axis >= fc * (1 - tol)) & (axis <= fc * (1 + tol))
        floor_mask = (np.abs(axis - fc) <= max(floor_halfwidth * f0, 3 * (axis[1] - axis[0]))) & ~peak_mask
        if not peak_mask.any() or floor_mask.sum() < 3:
            continue
        # A relative floor preserves the feature when sensor units change.
        epsilon = np.finfo(float).eps * spectrum.max()
        scores.append(np.log((spectrum[peak_mask].max() + epsilon) /
                             (np.median(spectrum[floor_mask]) + epsilon)))
    return float(np.mean(scores)) if scores else float("nan")


def bearing_cyclo_features(
    x,
    fs: float,
    fault_freqs: Mapping[str, float],
    harmonics: Sequence[int] = (1, 2, 3),
    tol: float = 0.03,
    nperseg: int = 64,
    hop: int = 8,
    band: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """Feature dict for classifiers and fusion: one IES score per fault frequency,
    plus band and impulsiveness descriptors."""
    if not fault_freqs:
        raise ValueError("supply named characteristic fault frequencies")
    x = _signal(x)
    f, _, P = stft_power(x, fs, nperseg, hop)
    sk = spectral_kurtosis_bands(P)
    alpha, ies = improved_envelope_spectrum(x, fs, band, nperseg, hop)
    out = {
        f"ies_score_{name}": fault_score(alpha, ies, f0, harmonics, tol)
        for name, f0 in fault_freqs.items()
    }
    out["sk_max"] = float(sk.max())
    out["sk_mean_positive"] = float(np.clip(sk, 0, None).mean())
    out["f_at_sk_max"] = float(f[int(np.argmax(sk))])
    variance = x.var()
    k = (((x - x.mean()) / np.sqrt(variance)) ** 4).mean() if variance > 0 else float("nan")
    out["kurtosis"] = float(k)
    return out


def max_fault_score(features: Mapping[str, float]) -> float:
    """Detection statistic: the strongest fault-frequency score (NaNs ignored)."""
    vals = [v for k, v in features.items() if k.startswith("ies_score_") and np.isfinite(v)]
    return float(max(vals)) if vals else float("nan")


def healthy_threshold(healthy_scores, alpha: float = 0.1, *, recording_ids=None,
                      split_manifest=None) -> float:
    """Conformal threshold on healthy max_fault_score values (one per recording).

    Healthy recordings must be disjoint from the evaluation specimens/recordings.
    Raises if there are too few healthy recordings for the requested alpha.
    Alarm only for score > threshold; equality is not an alarm. Supplying IDs
    and a manifest validates provenance; bare scores leave split checks to callers.
    """
    scores = np.asarray(healthy_scores, float)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("healthy scores must be a nonempty recording-level vector")
    if (recording_ids is None) != (split_manifest is None):
        raise ValueError("supply recording_ids and split_manifest together")
    if split_manifest is not None:
        from ..calibration.split_guard import canonical_manifest
        manifest = canonical_manifest(split_manifest)
        recording_ids = list(recording_ids)
        if len(recording_ids) != len(scores) or len(set(recording_ids)) != len(recording_ids):
            raise ValueError("use one score per unique independent recording")
        manifest.assert_development_units(recording_ids)
        groups = [group for unit in manifest.development_units for group in unit.group_ids]
        if len(groups) != len(set(groups)):
            raise ValueError("use one healthy calibration recording per independent specimen/asset")
    thr = conformal_quantile(scores, alpha)
    if not np.isfinite(thr):
        need = int(np.ceil(1.0 / alpha - 1))
        raise ValueError(f"need at least {need} healthy recordings for alpha={alpha}")
    return thr


# --------------------------------------------------------------------------- #
# Varying speed: order tracking
# --------------------------------------------------------------------------- #
def angular_resample(x, fs: float, speed_hz, samples_per_rev: int = 64,
                     *, anti_alias: bool = True) -> np.ndarray:
    """Resample x onto a uniform shaft-angle grid using the instantaneous speed.

    speed_hz: shaft frequency per sample (same length as x).  Apply this to a
    demodulated envelope (below), not to the raw high-frequency signal.
    """
    _positive(fs, "fs")
    _integer(samples_per_rev, "samples_per_rev", 4)
    x = _signal(x)
    speed = np.asarray(speed_hz, float)
    if x.shape != speed.shape or not np.isfinite(speed).all() or np.any(speed <= 0):
        raise ValueError("speed_hz must be finite, strictly positive and aligned with x")
    t = np.arange(x.size) / fs
    rev = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) / fs)])   # revolutions
    n_new = int(np.floor(rev[-1] * samples_per_rev)) + 1
    if n_new < 4:
        raise ValueError("record spans too little shaft angle to resample")
    # Conservative time-domain low-pass before angular downsampling. The minimum
    # speed defines the smallest output sampling rate along the record.
    if anti_alias and speed.min() * samples_per_rev < fs:
        cutoff = 0.45 * speed.min() * samples_per_rev
        x = sosfiltfilt(butter(4, cutoff, btype="lowpass", fs=fs, output="sos"), x)
    t_new = np.interp(np.arange(n_new) / samples_per_rev, rev, t)
    return np.interp(t_new, t, x)


def order_envelope_spectrum(
    x,
    fs: float,
    speed_hz,
    band: Tuple[float, float],
    samples_per_rev: int = 64,
    filter_order: int = 4,
):
    """Order-domain envelope spectrum: (orders, amplitude).

    Band-pass -> Hilbert envelope -> angular resampling -> FFT.  Fault orders are
    f_fault / f_shaft (e.g. BPFO order about 3.58 for a 9-ball bearing).
    """
    _positive(fs, "fs")
    _integer(filter_order, "filter_order")
    _integer(samples_per_rev, "samples_per_rev", 4)
    x = _signal(x)
    band = _band(band, fs, strict=True)
    sos = butter(filter_order, band, btype="bandpass", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x - x.mean())))
    env_ang = angular_resample(env, fs, speed_hz, samples_per_rev)
    if len(env_ang) < max(8, 2 * samples_per_rev):
        raise ValueError("order spectrum needs at least two shaft revolutions")
    env_ang = env_ang - env_ang.mean()
    win = np.hanning(env_ang.size)
    amp = np.abs(np.fft.rfft(env_ang * win)) * 2.0 / win.sum()
    orders = np.fft.rfftfreq(env_ang.size, 1.0 / samples_per_rev)
    return orders, amp


def order_fault_scores(orders, amp, fault_orders: Mapping[str, float],
                       harmonics: Sequence[int] = (1, 2, 3), tol: float = 0.02) -> Dict[str, float]:
    """Fault score per characteristic ORDER (same definition as `fault_score`)."""
    return {n: fault_score(orders, amp, o, harmonics, tol) for n, o in fault_orders.items()}
