import numpy as np

from tsdiag.domains.verification import (
    battery_physics_verification,
    transformer_physics_verification,
    turbofan_physics_verification,
)


def test_battery_verification_supports_consistent_localization():
    state = {
        "cell_voltage": np.array([[3.70, 3.52, 3.69], [3.70, 3.51, 3.69], [3.70, 3.50, 3.69]]),
        "cell_temperature": np.array([[30.0, 32.0, 30.1], [30.0, 32.2, 30.1], [30.0, 32.4, 30.1]]),
        "cell_ids": ["c0", "c1", "c2"],
        "top_cell": "c1",
        "delta_voltage": np.array([[0.01, -0.17, 0.00], [0.01, -0.18, 0.00], [0.01, -0.19, 0.00]]),
        "delta_temperature": np.array([[-0.1, 1.9, 0.0], [-0.1, 2.1, 0.0], [-0.1, 2.3, 0.0]]),
        "abnormal": True,
    }
    result = battery_physics_verification(state, "ev")
    assert result.status == "SUPPORTED"
    assert result.details["localization_consistent"] is True


def test_battery_verification_rejects_implausible_telemetry():
    state = {
        "cell_voltage": np.array([[8.0, 3.7], [8.0, 3.7]]),
        "cell_temperature": np.array([[30.0, 30.0], [30.0, 30.0]]),
        "cell_ids": ["c0", "c1"],
        "top_cell": "c0",
        "delta_voltage": np.array([[2.15, -2.15]] * 2),
        "delta_temperature": np.zeros((2, 2)),
        "abnormal": True,
    }
    assert battery_physics_verification(state, "ev").status == "CONTRADICTED"


def test_turbofan_verification_supports_monotonic_degradation():
    cycles = np.arange(1, 21, dtype=float)
    health = np.linspace(0.0, 2.0, len(cycles))
    state = {
        "cycle_index": cycles,
        "health_index": health,
        "health_slope": 0.1,
        "rul_cycles": 15.0,
        "abnormal": True,
    }
    result = turbofan_physics_verification(state, "ev")
    assert result.status == "SUPPORTED"
    assert result.details["health_cycle_correlation"] > 0.9


def test_transformer_verification_requires_multisensor_support():
    state = {
        "harmonic_structure": {"anomaly_score": 0.7, "kurtosis": 7.0},
        "sensor_weights": np.array([0.6, 0.4]),
        "channel_statistics": {"std": np.array([1.0, 0.8])},
        "predicted_fault": "winding_fault",
        "abnormal": True,
        "anomaly_threshold": 0.25,
    }
    assert transformer_physics_verification(state, "ev").status == "SUPPORTED"

    state["channel_statistics"] = {"std": np.array([1.0, 0.0])}
    assert transformer_physics_verification(state, "ev").status == "INSUFFICIENT"


def test_transformer_verification_contradicts_unbacked_classifier_label():
    state = {
        "harmonic_structure": {"anomaly_score": 0.05, "kurtosis": 3.0},
        "sensor_weights": np.array([0.5, 0.5]),
        "channel_statistics": {"std": np.array([1.0, 1.0])},
        "predicted_fault": "winding_fault",
        "abnormal": False,
        "anomaly_threshold": 0.25,
    }
    assert transformer_physics_verification(state, "ev").status == "CONTRADICTED"
