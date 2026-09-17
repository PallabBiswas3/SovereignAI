import numpy as np

from tsdiag.tools import (
    change_point_detection,
    correlation_sensor_weighting,
    cross_correlation_analysis,
    default_tool_registry,
    multisensor_fusion,
    pca_monitoring,
    signal_integrity,
    spectral_kurtosis,
    standardize_against_normal,
    stationarity_analysis,
    time_domain_features,
    wavelet_denoising,
)


def test_signal_features_and_integrity():
    fs = 2000.0
    t = np.arange(0, 2, 1 / fs)
    x = np.sin(2 * np.pi * 120 * t)
    quality = signal_integrity(x, fs)
    feats = time_domain_features(x)
    assert quality["quality_flags"] == []
    assert feats["rms"] > 0
    assert 1.0 < feats["kurtosis"] < 2.0


def test_spectral_kurtosis_returns_band():
    fs = 4000.0
    t = np.arange(0, 2, 1 / fs)
    carrier = np.sin(2 * np.pi * 900 * t)
    impulses = (np.sin(2 * np.pi * 35 * t) > 0.97).astype(float)
    x = carrier * impulses + 0.05 * np.random.default_rng(1).normal(size=t.size)
    result = spectral_kurtosis(x, fs)
    assert result["recommended_band_hz"] is not None
    low, high = result["recommended_band_hz"]
    assert 0 < low < high < fs / 2


def test_pca_monitoring_flags_shifted_samples():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(1000, 3))
    cur = rng.normal(size=(200, 3)) + np.array([4.0, 0.0, 0.0])
    standardized = standardize_against_normal(cur, ref)
    result = pca_monitoring(standardized["standardized_signal"], standardized["standardized_reference"])
    assert result["alarm_mask"].mean() > 0.5


def test_stationarity_detects_random_walk_as_nonstationary():
    rng = np.random.default_rng(3)
    random_walk = np.cumsum(rng.normal(size=800))
    stationary = rng.normal(size=800)
    result = stationarity_analysis(np.column_stack([random_walk, stationary]))
    statuses = [x["status"] for x in result["stationarity_flags"]]
    assert "nonstationary" in statuses


def test_change_point_detector_finds_mean_shift():
    rng = np.random.default_rng(4)
    x = np.r_[rng.normal(0, 0.2, 200), rng.normal(2.5, 0.2, 200)]
    result = change_point_detection(x, min_size=25, z_threshold=4.0)
    assert result["change_points"]
    assert abs(result["change_points"][0]["index"] - 200) < 40


def test_wavelet_and_multisensor_fusion():
    rng = np.random.default_rng(5)
    t = np.linspace(0, 1, 1024, endpoint=False)
    clean = np.sin(2 * np.pi * 30 * t)
    matrix = np.column_stack([clean + 0.2 * rng.normal(size=t.size), clean + 0.2 * rng.normal(size=t.size)])
    denoised = wavelet_denoising(matrix)["denoised_signal_matrix"]
    corr = cross_correlation_analysis(denoised)
    weights = correlation_sensor_weighting(corr["correlation_energy"])["sensor_weights"]
    fused = multisensor_fusion(denoised, weights)["fused_waveform"]
    assert denoised.shape == matrix.shape
    assert np.isclose(weights.sum(), 1.0)
    assert fused.shape == (matrix.shape[0],)


def test_registry_covers_new_shared_tools():
    registry = default_tool_registry()
    for name in [
        "signal_integrity",
        "spectral_kurtosis",
        "stationarity_analysis",
        "change_point_detection",
        "wavelet_denoising",
        "pca_monitoring",
    ]:
        assert registry.get(name) is not None
