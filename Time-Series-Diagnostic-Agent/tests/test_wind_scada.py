from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from tsdiag.datasets.wind_care import (
    care_normal_mask,
    load_care_event,
    load_care_event_asset_ids,
    validate_care_layout,
)
from tsdiag.domains.wind_scada_runner import WindScadaDiagnosticPipeline
from tsdiag.tools.wind_scada import (
    WindNormalBehaviorState,
    detect_wind_operating_regimes,
    fit_wind_normal_behavior_model,
    predict_wind_normal_behavior,
    scada_quality_check,
    wind_residual_anomaly_detection,
)


def _write_care_fixture(root: Path):
    for wf in ("A", "B", "C"):
        folder = root / f"Wind Farm {wf}"
        (folder / "datasets").mkdir(parents=True)
        pd.DataFrame(
            [{
                "event_id": 1 if wf == "A" else 100 + ord(wf),
                "asset_id": 0,
                "event_start": "2022-01-02",
                "event_end": "2022-01-03",
                "event_label": "anomaly" if wf == "A" else "normal",
            }]
        ).to_csv(folder / "event_info.csv", sep=";", index=False)
        pd.DataFrame(
            [
                {"sensor_name": "wind_speed", "statistics_type": "average"},
                {"sensor_name": "power", "statistics_type": "average"},
                {"sensor_name": "temp", "statistics_type": "average"},
            ]
        ).to_csv(folder / "feature_description.csv", sep=";", index=False)

    frame = pd.DataFrame(
        {
            "id": range(8),
            "train_test": ["train"] * 5 + ["prediction"] * 3,
            "time_stamp": pd.date_range("2022-01-01", periods=8, freq="10min"),
            "asset_id": [0] * 8,
            "status_type_id": [0, 0, 0, 1, 0, 0, 4, 0],
            "wind_speed": np.linspace(4, 11, 8),
            "power": np.linspace(50, 900, 8),
            "temp": np.linspace(30, 40, 8),
        }
    )
    frame.to_csv(root / "Wind Farm A" / "datasets" / "1.csv", sep=";", index=False)


def test_care_loader_uses_train_prediction_split(tmp_path):
    _write_care_fixture(tmp_path)
    layout = validate_care_layout(tmp_path)
    assert layout == {"A": True, "B": True, "C": True}

    event = load_care_event(tmp_path, 1)
    assert event.event.is_anomaly is True
    assert len(event.train) == 5
    assert len(event.prediction) == 3
    assert {"wind_speed", "power", "temp"}.issubset(event.train.columns)
    assert care_normal_mask(event.train).tolist() == [True, True, True, False, True]
    assert load_care_event_asset_ids(tmp_path, 1) == ("0",)


def test_care_loader_streams_official_style_zip(tmp_path):
    fixture = tmp_path / "care"
    _write_care_fixture(fixture)
    archive = Path(shutil.make_archive(str(tmp_path / "CARE_To_Compare"), "zip", fixture))
    assert validate_care_layout(archive) == {"A": True, "B": True, "C": True}
    event = load_care_event(archive, 1)
    assert event.event.is_anomaly is True
    assert len(event.train) == 5
    assert len(event.prediction) == 3
    assert load_care_event_asset_ids(archive, 1) == ("0",)


def test_scada_quality_check_detects_irregular_sampling():
    x = np.column_stack([np.arange(5.0), np.ones(5)])
    ts = np.array([
        "2022-01-01T00:00",
        "2022-01-01T00:10",
        "2022-01-01T00:20",
        "2022-01-01T00:50",
        "2022-01-01T01:00",
    ], dtype="datetime64[m]")
    result = scada_quality_check(x, ["wind", "constant"], ts)
    assert result["sampling_irregularity"] > 0
    assert any(flag.get("flag") == "constant_channel" for flag in result["quality_flags"])


def test_regime_aware_normal_behavior_flags_persistent_shift():
    rng = np.random.default_rng(4)
    n_ref = 500
    wind_ref = rng.uniform(3, 14, n_ref)
    regime_ref = (wind_ref >= 8).astype(int)
    power_ref = np.where(regime_ref == 0, 15 * wind_ref**2, 900 + 5 * wind_ref) + rng.normal(0, 15, n_ref)
    temp_ref = 25 + 0.015 * power_ref + rng.normal(0, 0.5, n_ref)
    ref = np.column_stack([wind_ref, power_ref, temp_ref])

    current = ref[:160].copy()
    current_regime = regime_ref[:160].copy()
    current[100:, 2] += 8.0

    state = fit_wind_normal_behavior_model(
        ref,
        regime_ref,
        target_indices=[2],
        predictor_indices=[0, 1, 2],
    )
    predicted = predict_wind_normal_behavior(current, current_regime, state)
    detected = wind_residual_anomaly_detection(
        predicted["normalized_residuals"],
        threshold=3.0,
        persistence=3,
    )

    assert detected["alarm_mask"][:90].mean() < 0.10
    assert detected["alarm_mask"][110:].mean() > 0.50


def test_residual_persistence_does_not_chain_across_different_channels():
    residuals = np.zeros((6, 3), dtype=float)
    residuals[1, 0] = 5.0
    residuals[2, 1] = 5.0
    residuals[3, 2] = 5.0

    detected = wind_residual_anomaly_detection(
        residuals,
        threshold=3.5,
        persistence=3,
    )

    assert detected["raw_alarm_mask"][1:4].tolist() == [True, True, True]
    assert not np.any(detected["alarm_mask"])


def test_residual_persistence_detects_same_channel_run():
    residuals = np.zeros((6, 3), dtype=float)
    residuals[1:4, 1] = 5.0

    detected = wind_residual_anomaly_detection(
        residuals,
        threshold=3.5,
        persistence=3,
    )

    assert detected["alarm_mask"][3]
    assert detected["persistent_channel_alarm_mask"][3, 1]


def test_out_of_sample_residual_scale_stays_sane_on_healthy_holdout():
    rng = np.random.default_rng(17)
    n_ref = 800
    wind_ref = rng.uniform(3.0, 14.0, n_ref)
    power_ref = 22.0 * wind_ref**2 + rng.normal(0.0, 30.0, n_ref)
    temp_ref = 20.0 + 0.012 * power_ref + np.sin(wind_ref) + rng.normal(0.0, 0.7, n_ref)
    ref = np.column_stack([wind_ref, power_ref, temp_ref])
    regimes = (wind_ref >= 8.0).astype(int)

    n_test = 250
    wind_test = rng.uniform(3.0, 14.0, n_test)
    power_test = 22.0 * wind_test**2 + rng.normal(0.0, 30.0, n_test)
    temp_test = 20.0 + 0.012 * power_test + np.sin(wind_test) + rng.normal(0.0, 0.7, n_test)
    test = np.column_stack([wind_test, power_test, temp_test])
    test_regimes = (wind_test >= 8.0).astype(int)

    state = fit_wind_normal_behavior_model(
        ref,
        regimes,
        target_indices=[2],
        predictor_indices=[0, 1, 2],
    )
    prediction = predict_wind_normal_behavior(test, test_regimes, state)
    z = np.abs(prediction["normalized_residuals"][:, 0])
    assert np.nanmedian(z) < 1.5
    assert np.mean(z >= 3.5) < 0.10


def test_oos_residual_calibration_recenters_systematic_healthy_model_bias():
    rng = np.random.default_rng(23)
    n_ref = 700
    wind = rng.uniform(3.0, 14.0, n_ref)
    power = 18.0 * wind**2 + rng.normal(0.0, 20.0, n_ref)
    temp = 24.0 + 0.01 * power + 0.5 * np.sin(1.7 * wind) + rng.normal(0.25, 0.8, n_ref)
    ref = np.column_stack([wind, power, temp])
    regimes = (wind >= 8.0).astype(int)

    state = fit_wind_normal_behavior_model(
        ref,
        regimes,
        target_indices=[2],
        predictor_indices=[0, 1, 2],
    )

    prediction = predict_wind_normal_behavior(ref, regimes, state)
    z = prediction["normalized_residuals"][:, 0]
    assert np.isfinite(state.residual_center[0])
    assert abs(np.nanmedian(z)) < 0.5


def test_fit_records_finite_per_regime_residual_calibration():
    rng = np.random.default_rng(24)
    n = 1000
    regime = np.r_[np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)]
    wind = np.r_[rng.uniform(3.0, 7.0, n // 2), rng.uniform(9.0, 14.0, n // 2)]
    power = np.where(regime == 0, 17.0 * wind**2, 760.0 + 9.0 * wind) + rng.normal(0.0, 18.0, n)
    temp = 20.0 + 0.012 * power + np.where(regime == 0, -1.5, 2.0) + rng.normal(0.0, 0.7, n)
    ref = np.column_stack([wind, power, temp])

    state = fit_wind_normal_behavior_model(
        ref,
        regime,
        target_indices=[2],
        predictor_indices=[0, 1, 2],
    )

    assert set(state.residual_center_by_regime) == {0, 1}
    assert set(state.residual_scale_by_regime) == {0, 1}
    assert np.isfinite(state.residual_center_by_regime[0][0])
    assert np.isfinite(state.residual_center_by_regime[1][0])
    assert state.residual_scale_by_regime[0][0] > 0.0
    assert state.residual_scale_by_regime[1][0] > 0.0


def test_prediction_uses_regime_specific_residual_center_and_scale():
    class ZeroModel:
        def predict(self, x):
            return np.zeros(len(x), dtype=float)

    state = WindNormalBehaviorState(
        target_indices=[1],
        predictor_indices=[0],
        models={(0, 1): ZeroModel(), (1, 1): ZeroModel()},
        global_models={1: ZeroModel()},
        residual_center=np.array([0.0]),
        residual_scale=np.array([10.0]),
        residual_center_by_regime={0: np.array([1.0]), 1: np.array([5.0])},
        residual_scale_by_regime={0: np.array([2.0]), 1: np.array([4.0])},
        regime_ids=[0, 1],
    )
    x = np.array([
        [0.0, 3.0],
        [0.0, 9.0],
    ])
    prediction = predict_wind_normal_behavior(x, np.array([0, 1]), state)

    assert np.allclose(prediction["normalized_residuals"][:, 0], [1.0, 1.0])
    assert np.allclose(prediction["residual_center_used"][:, 0], [1.0, 5.0])
    assert np.allclose(prediction["residual_scale_used"][:, 0], [2.0, 4.0])


def test_operating_regime_detection_returns_requested_number():
    rng = np.random.default_rng(8)
    x = np.r_[
        rng.normal([4, 100], [0.3, 10], size=(80, 2)),
        rng.normal([8, 600], [0.3, 20], size=(80, 2)),
        rng.normal([12, 1000], [0.3, 15], size=(80, 2)),
    ]
    result = detect_wind_operating_regimes(x, driver_indices=[0, 1], n_regimes=3)
    assert len(np.unique(result["regime_ids"])) == 3


def test_regime_driver_selection_prefers_active_power_over_reactive_power():
    rng = np.random.default_rng(31)
    names = [
        "Wind Speed Avg",
        "Reactive Power Avg",
        "Active Power Avg",
        "Rotor Speed Avg",
        "Generator Speed Avg",
        "Main Bearing Temp Avg",
    ]
    matrix = rng.normal(size=(300, len(names)))
    matrix[:, 1] *= 1000.0
    matrix[:, 2] *= 100.0

    drivers, roles = WindScadaDiagnosticPipeline._driver_selection(names, matrix)

    assert roles["active_power"] == 2
    assert 1 not in drivers
    assert names[roles["active_power"]] == "Active Power Avg"


def test_regime_assignment_diagnostics_flag_far_out_of_support_operation():
    rng = np.random.default_rng(32)
    n = 500
    wind = rng.uniform(4.0, 12.0, n)
    active_power = 20.0 * wind**2 + rng.normal(0.0, 20.0, n)
    rotor_speed = 8.0 * wind + rng.normal(0.0, 1.0, n)
    temp = 25.0 + 0.01 * active_power + rng.normal(0.0, 0.5, n)
    train = np.column_stack([wind, active_power, rotor_speed, temp])

    prediction = train[:120].copy()
    prediction[:, 0] += 30.0
    prediction[:, 1] += 5000.0
    names = ["Wind Speed", "Active Power", "Rotor Speed", "Main Bearing Temp"]

    result = WindScadaDiagnosticPipeline(n_regimes=3).run(
        train,
        prediction,
        names,
        target_indices=[3],
    )
    diagnostics = result.artifacts["regime_assignment"]

    assert diagnostics["prediction_nearest_distance"].shape == (len(prediction),)
    assert diagnostics["out_of_distribution_mask"].shape == (len(prediction),)
    assert diagnostics["out_of_distribution_fraction"] > 0.50
    assert result.artifacts["driver_roles"]["active_power"]["channel"] == "Active Power"
