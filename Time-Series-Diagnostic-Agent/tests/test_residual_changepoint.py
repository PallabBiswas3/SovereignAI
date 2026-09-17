import numpy as np

from tsdiag.detectors.residual_changepoint import (
    DEFAULT_CUSUM_DRIFT,
    DEFAULT_CUSUM_HOLD_SAMPLES,
    DEFAULT_CUSUM_THRESHOLD,
    ResidualCUSUMConfig,
    residual_cusum,
)


def test_canonical_cusum_defaults_are_consistent():
    config = ResidualCUSUMConfig()
    assert config.drift == DEFAULT_CUSUM_DRIFT == 0.5
    assert config.threshold == DEFAULT_CUSUM_THRESHOLD == 10.0
    assert config.hold_samples == DEFAULT_CUSUM_HOLD_SAMPLES == 6


def test_cusum_detects_low_amplitude_persistent_shift():
    rng = np.random.default_rng(11)
    residuals = rng.normal(0.0, 0.20, size=(240, 3))
    residuals[120:, 1] += 0.85
    result = residual_cusum(
        residuals,
        config=ResidualCUSUMConfig(drift=0.20, threshold=5.0, hold_samples=5),
    )
    assert result["change_points"]
    assert result["change_points"][0] >= 120
    assert result["change_points"][0] < 160
    assert 1 in result["channel_change_points"]
    assert np.any(result["alarm_mask"])


def test_cusum_stays_quiet_on_small_centered_noise():
    rng = np.random.default_rng(12)
    residuals = rng.normal(0.0, 0.08, size=(300, 2))
    result = residual_cusum(
        residuals,
        config=ResidualCUSUMConfig(drift=0.20, threshold=8.0, hold_samples=4),
    )
    assert result["change_points"] == []
    assert not np.any(result["alarm_mask"])


def test_multichannel_consensus_ignores_isolated_channel_shift_but_keeps_localization():
    rng = np.random.default_rng(21)
    residuals = rng.normal(0.0, 0.12, size=(260, 4))
    residuals[120:, 2] += 0.95
    result = residual_cusum(
        residuals,
        config=ResidualCUSUMConfig(
            drift=0.20,
            threshold=5.0,
            hold_samples=5,
            min_channel_support=2,
        ),
    )
    assert 2 in result["channel_change_points"]
    assert result["change_points"] == []
    assert not np.any(result["alarm_mask"])


def test_multichannel_consensus_detects_coordinated_persistent_shift():
    rng = np.random.default_rng(22)
    residuals = rng.normal(0.0, 0.12, size=(260, 4))
    residuals[120:, 1] += 0.95
    residuals[120:, 3] += 0.90
    result = residual_cusum(
        residuals,
        config=ResidualCUSUMConfig(
            drift=0.20,
            threshold=5.0,
            hold_samples=5,
            min_channel_support=2,
        ),
    )
    assert result["change_points"]
    assert result["change_points"][0] >= 120
    assert result["change_points"][0] < 170
    assert np.max(result["channel_support_count"]) >= 2
    assert np.any(result["alarm_mask"])
