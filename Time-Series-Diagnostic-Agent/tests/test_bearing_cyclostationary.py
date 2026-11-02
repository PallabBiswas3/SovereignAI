import numpy as np
import pytest

from tsdiag.challengers.bearing_cyclostationary import (
    angular_resample,
    bearing_cyclo_features,
    cyclic_modulation_depth,
    fault_score,
    healthy_threshold,
    improved_envelope_spectrum,
    max_fault_score,
    order_envelope_spectrum,
    order_fault_scores,
    select_resonance_band,
    spectral_kurtosis_bands,
    stft_power,
)

FS = 12000.0
FAULTS = {"BPFO": 85.0, "BPFI": 123.0}


def _signal(rng, n=24000, fault=None, snr=1.5, f_res=3000.0, tonal=True, jitter=0.01):
    t = np.arange(n) / FS
    x = rng.normal(size=n)
    if tonal:
        x = x + 1.5 * np.sin(2 * np.pi * 1000.0 * t)               # gear-mesh-like tone
    if fault is not None:
        f0 = FAULTS[fault]
        k = int(0.03 * FS)
        tau = np.arange(k) / FS
        ring = np.exp(-tau * 800.0) * np.sin(2 * np.pi * f_res * tau)
        imp = np.zeros(n)
        pos = 0.0
        while pos < n - k:
            imp[int(pos): int(pos) + k] += ring * rng.uniform(0.7, 1.3)
            pos += FS / f0 * (1 + rng.normal(scale=jitter))         # random slip
        x = x + snr * imp
    return x


def test_noise_has_near_zero_spectral_kurtosis():
    rng = np.random.default_rng(0)
    _, _, P = stft_power(rng.normal(size=24000), FS)
    assert abs(spectral_kurtosis_bands(P).mean()) < 0.2


def test_resonance_band_is_found():
    rng = np.random.default_rng(1)
    lo, hi = select_resonance_band(_signal(rng, fault="BPFO", snr=1.0), FS, bandwidth_hz=1000)
    assert lo - 300 <= 3000 <= hi + 300


def test_ies_peaks_at_the_right_fault_frequency():
    rng = np.random.default_rng(2)
    x = _signal(rng, fault="BPFO")
    alpha, ies = improved_envelope_spectrum(x, FS)
    sel = (alpha > 60) & (alpha < 110)
    assert abs(alpha[sel][np.argmax(ies[sel])] - FAULTS["BPFO"]) < 3.0
    assert fault_score(alpha, ies, FAULTS["BPFO"]) > fault_score(alpha, ies, FAULTS["BPFI"]) + 0.5


def test_fault_discrimination_bpfo_vs_bpfi():
    rng = np.random.default_rng(3)
    fo = bearing_cyclo_features(_signal(rng, fault="BPFO"), FS, FAULTS)
    fi = bearing_cyclo_features(_signal(rng, fault="BPFI"), FS, FAULTS)
    assert fo["ies_score_BPFO"] > fo["ies_score_BPFI"]
    assert fi["ies_score_BPFI"] > fi["ies_score_BPFO"]
    assert abs(fo["f_at_sk_max"] - 3000) < 600


def test_healthy_threshold_controls_false_alarms_and_detects_faults():
    rng = np.random.default_rng(4)
    healthy = [max_fault_score(bearing_cyclo_features(_signal(rng), FS, FAULTS)) for _ in range(60)]
    thr = healthy_threshold(healthy, alpha=0.1)
    fresh = [max_fault_score(bearing_cyclo_features(_signal(rng), FS, FAULTS)) for _ in range(60)]
    assert np.mean(np.array(fresh) > thr) <= 0.2                  # target 0.1, 60 recordings
    faulty = [max_fault_score(bearing_cyclo_features(_signal(rng, fault="BPFO"), FS, FAULTS))
              for _ in range(30)]
    assert np.mean(np.array(faulty) > thr) >= 0.9


def test_healthy_threshold_needs_enough_recordings():
    with pytest.raises(ValueError, match="at least"):
        healthy_threshold(np.arange(5.0), alpha=0.1)


def test_modulation_depth_shapes():
    rng = np.random.default_rng(5)
    f, rate, P = stft_power(rng.normal(size=12000), FS)
    alpha, gamma = cyclic_modulation_depth(P, rate)
    assert gamma.shape == (P.shape[0], alpha.size) and np.all(gamma >= 0)


# ---- varying speed -------------------------------------------------------------
def test_angular_resample_uniform_in_angle():
    fs = 1000.0
    n = 20000
    speed = np.linspace(10.0, 30.0, n)                              # 10 -> 30 Hz ramp
    t = np.arange(n) / fs
    rev = np.cumsum(speed) / fs
    x = np.sin(2 * np.pi * 3.0 * rev)                               # 3 cycles per revolution
    xa = angular_resample(x, fs, speed, samples_per_rev=64)
    spec = np.abs(np.fft.rfft(xa * np.hanning(xa.size)))
    orders = np.fft.rfftfreq(xa.size, 1 / 64)
    assert abs(orders[np.argmax(spec)] - 3.0) < 0.05


def test_order_tracking_finds_fault_order_under_speed_ramp():
    rng = np.random.default_rng(6)
    fs, n = 12000.0, 12000 * 4
    speed = np.linspace(20.0, 30.0, n)                              # shaft Hz, varying
    rev = np.cumsum(speed) / fs
    order_fault = 3.58                                              # BPFO order
    k = int(0.02 * fs)
    tau = np.arange(k) / fs
    ring = np.exp(-tau * 900.0) * np.sin(2 * np.pi * 3000.0 * tau)
    x = rng.normal(size=n)
    next_rev = 0.0
    while True:
        idx = int(np.searchsorted(rev, next_rev))
        if idx >= n - k:
            break
        x[idx: idx + k] += 2.5 * ring
        next_rev += 1.0 / order_fault
    orders, amp = order_envelope_spectrum(x, fs, speed, band=(2000, 4000))
    sel = (orders > 2.5) & (orders < 5.0)
    assert abs(orders[sel][np.argmax(amp[sel])] - order_fault) < 0.1
    scores = order_fault_scores(orders, amp, {"BPFO": order_fault, "OTHER": 5.3}, tol=0.02)
    assert scores["BPFO"] > scores["OTHER"] + 1.0


def test_angular_resample_validates_lengths():
    with pytest.raises(ValueError):
        angular_resample(np.zeros(10), 1000.0, np.ones(9))


def test_dimensionless_features_do_not_change_with_sensor_amplitude_units():
    sample = _signal(np.random.default_rng(20), fault="BPFO")
    original = bearing_cyclo_features(sample, FS, FAULTS)
    scaled = bearing_cyclo_features(sample * 1e-6, FS, FAULTS)
    for name in original:
        assert scaled[name] == pytest.approx(original[name], rel=1e-7, abs=1e-9)


@pytest.mark.parametrize("kwargs", [{"fs": 0}, {"fs": np.nan}, {"nperseg": 63},
                                    {"hop": 0}, {"hop": 65}])
def test_stft_configuration_is_validated(kwargs):
    arguments = {"fs": FS, **kwargs}
    with pytest.raises(ValueError):
        stft_power(np.zeros(200), **arguments)


def test_short_or_missing_waveforms_are_not_silently_repaired():
    with pytest.raises(ValueError):
        stft_power(np.zeros(20), FS)
    with pytest.raises(ValueError):
        stft_power(np.full(200, np.nan), FS)


@pytest.mark.parametrize("power", [np.ones((4, 3)), np.full((4, 10), -1.0),
                                  np.full((4, 10), np.nan), np.ones(10)])
def test_invalid_power_matrices_are_rejected(power):
    with pytest.raises(ValueError):
        spectral_kurtosis_bands(power)
    with pytest.raises(ValueError):
        cyclic_modulation_depth(power, 100)


def test_empty_out_of_range_and_unscorable_resonance_bands():
    sample = _signal(np.random.default_rng(21))
    with pytest.raises(ValueError, match="no resolved"):
        improved_envelope_spectrum(sample, FS, band=(1, 2))
    with pytest.raises(ValueError, match="Nyquist"):
        improved_envelope_spectrum(sample, FS, band=(5000, 7000))
    with pytest.raises(ValueError, match="bandwidth"):
        select_resonance_band(sample, FS, bandwidth_hz=10000)
    with pytest.raises(ValueError, match="silent"):
        select_resonance_band(np.zeros(12000), FS)


def test_zero_energy_abstains_and_flat_spectra_have_no_positive_fault_evidence():
    features = bearing_cyclo_features(np.zeros(12000), FS, FAULTS)
    assert np.isnan(max_fault_score(features))
    axis = np.arange(1001.0)
    assert fault_score(axis, np.ones_like(axis), 100) == 0


@pytest.mark.parametrize("kind", ["unsorted", "negative", "nan", "mismatch", "frequency"])
def test_fault_scores_reject_malformed_spectra(kind):
    axis, spectrum, f0 = np.arange(1001.0), np.ones(1001), 100
    if kind == "unsorted": axis[10] = 0
    if kind == "negative": spectrum[10] = -1
    if kind == "nan": spectrum[10] = np.nan
    if kind == "mismatch": spectrum = spectrum[:-1]
    if kind == "frequency": f0 = 0
    with pytest.raises(ValueError):
        fault_score(axis, spectrum, f0)


def test_healthy_threshold_uses_existing_manifest_and_independent_recordings():
    from tsdiag.calibration import SplitManifest
    ids = tuple(f"record-{i}" for i in range(9))
    assets = {i: f"specimen-{i}" for i in (*ids, "evaluation")}
    manifest = SplitManifest("asset", ids, ("evaluation",), assets)
    assert healthy_threshold(np.arange(9.0), recording_ids=ids, split_manifest=manifest) == 8
    with pytest.raises(ValueError, match="unique"):
        healthy_threshold(np.arange(9.0), recording_ids=[ids[0]] * 9, split_manifest=manifest)
    with pytest.raises(ValueError):
        healthy_threshold(np.arange(9.0), recording_ids=[*ids[:-1], "evaluation"], split_manifest=manifest)
    assets = {i: "same-specimen" for i in ids}
    assets["evaluation"] = "separate-specimen"
    with pytest.raises(ValueError, match="independent specimen"):
        healthy_threshold(np.arange(9.0), recording_ids=ids,
                          split_manifest=SplitManifest("asset", ids, ("evaluation",), assets))


@pytest.mark.parametrize("speed", [np.zeros(100), -np.ones(100), np.full(100, np.nan)])
def test_order_tracking_requires_a_valid_positive_tachometer(speed):
    with pytest.raises(ValueError, match="speed_hz"):
        angular_resample(np.ones(100), 1000, speed)


def test_insufficient_shaft_rotation_and_invalid_filter_band_fail_explicitly():
    with pytest.raises(ValueError, match="shaft angle"):
        angular_resample(np.zeros(100), FS, np.full(100, 0.01))
    with pytest.raises(ValueError, match="Nyquist"):
        order_envelope_spectrum(np.ones(100), 1000, np.ones(100), band=(0, 200))
