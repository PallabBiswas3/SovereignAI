import json

import numpy as np
import pytest

import tsdiag.evaluation.wind_calibration as calibration_module
from tsdiag.evaluation.wind_calibration import (
    WindCalibrationCandidate,
    WindCalibrationResult,
    _PreparedHealthyEvent,
    _split_healthy_window,
    calibrate_wind_residual_config,
    load_wind_calibrated_config,
    save_wind_calibration_result,
)
from tsdiag.evaluation.split_manifest import DevelopmentSplitManifest, SplitUnit


def _event(event_id: int) -> dict:
    return {
        "event_id": event_id,
        "train_matrix": np.zeros((100, 2)),
        "healthy_train_mask": np.ones(100, dtype=bool),
        "channel_names": ["power", "wind_speed"],
    }


def _manifest(development_ids, evaluation_ids):
    return DevelopmentSplitManifest(
        dataset_id="care-v6:test",
        development_pool_id="care-development-v1",
        evaluation_pool_id="care-frozen-v1",
        leakage_boundary="unit",
        unit_kind="care_event",
        group_kind=None,
        development_units=tuple(SplitUnit(str(value)) for value in development_ids),
        evaluation_units=tuple(SplitUnit(str(value)) for value in evaluation_ids),
    )


def test_healthy_split_uses_longest_contiguous_window_and_tail_holdout():
    matrix = np.arange(240, dtype=float).reshape(120, 2)
    mask = np.zeros(120, dtype=bool)
    mask[5:25] = True
    mask[30:110] = True
    fit, holdout = _split_healthy_window(
        matrix,
        mask,
        holdout_fraction=0.25,
        minimum_partition_samples=10,
    )
    assert np.array_equal(fit, matrix[30:90])
    assert np.array_equal(holdout, matrix[90:110])


def test_calibration_rejects_development_benchmark_overlap():
    with pytest.raises(ValueError, match="overlap"):
        calibrate_wind_residual_config(
            [_event(7)],
            split_manifest=DevelopmentSplitManifest(
                dataset_id="care-v6:test",
                development_pool_id="care-development-v1",
                evaluation_pool_id="care-frozen-v1",
                leakage_boundary="unit",
                unit_kind="care_event",
                group_kind=None,
                development_units=(SplitUnit("7"),),
                evaluation_units=(SplitUnit("7"), SplitUnit("8")),
            ),
        )


def test_calibration_selects_loosest_passing_candidate(monkeypatch):
    def prepared(event, **kwargs):
        return _PreparedHealthyEvent(
            event_id=event["event_id"],
            normalized_residuals=np.zeros((40, 2)),
            regime_transition_mask=np.zeros(40, dtype=bool),
        )

    monkeypatch.setattr(calibration_module, "_prepare_healthy_event", prepared)
    result = calibrate_wind_residual_config(
        [_event(1), _event(2), _event(3)],
        split_manifest=_manifest((1, 2, 3), range(50, 60)),
        target_event_far=0.08,
        residual_threshold_grid=(3.0, 5.0),
        persistence_grid=(2, 4),
        cusum_drift_grid=(0.5,),
        cusum_threshold_grid=(10.0, 20.0),
        event_minimum_residual_fraction_grid=(0.02, 0.08),
        bootstrap_samples=20,
    )
    assert result.met_target is True
    assert result.achieved_event_far == 0.0
    assert result.config == WindCalibrationCandidate(3.0, 2, 0.5, 10.0, 0.02)


def test_frozen_config_round_trip_and_unmet_rejection(tmp_path):
    result = WindCalibrationResult(
        config=WindCalibrationCandidate(4.0, 3, 0.75, 14.0, 0.04),
        achieved_event_far=0.05,
        achieved_event_far_ci=(0.0, 0.12),
        achieved_sample_far=0.03,
        target_event_far=0.08,
        n_holdout_samples=500,
        n_events_used=20,
        event_ids=tuple(range(20)),
        development_pool_id="independent-healthy-v1",
        evaluation_pool_id="care-frozen-v1",
        leakage_boundary="unit",
        split_manifest_sha256="abc123",
        met_target=True,
    )
    path = save_wind_calibration_result(result, tmp_path / "wind.json")
    loaded = load_wind_calibrated_config(path)
    assert loaded["config"]["cusum_threshold"] == 14.0

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["met_target"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="did not meet"):
        load_wind_calibrated_config(path)


def test_wind_plugin_loads_accepted_calibration_by_default(monkeypatch):
    import tsdiag.domains.wind_scada_plugin as plugin_module
    from tsdiag.contracts import DiagnosticRequest

    payload = {
        "development_pool_id": "independent-healthy-v1",
        "target_event_far": 0.08,
        "achieved_event_far": 0.05,
        "achieved_event_far_ci": [0.01, 0.12],
        "n_events_used": 20,
        "config": {
            "residual_threshold": 4.5,
            "persistence": 4,
            "cusum_drift": 0.75,
            "cusum_threshold": 14.0,
            "event_minimum_residual_fraction": 0.04,
        },
    }
    captured = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, train_matrix, prediction_matrix, channel_names, **kwargs):
            n = len(prediction_matrix)
            mask = np.zeros(n, dtype=bool)
            return type("Result", (), {
                "affected_channels": [],
                "confidence": 0.0,
                "alarm_mask": mask,
                "anomaly_scores": np.zeros(n),
                "artifacts": {
                    "anomaly_detection": {"alarm_mask": mask},
                    "residual_changepoint": {"alarm_mask": mask},
                    "regime_assignment": {"out_of_distribution_mask": mask},
                    "physics_consistency": {"verification_findings": []},
                },
            })()

    monkeypatch.setattr(plugin_module, "load_wind_calibrated_config", lambda: payload)
    monkeypatch.setattr(plugin_module, "WindScadaDiagnosticPipeline", FakePipeline)
    plugin = plugin_module.WindScadaPlugin()
    request = DiagnosticRequest(
        domain="wind_scada",
        task="condition_monitoring",
        inputs={
            "train_matrix": np.ones((80, 2)),
            "prediction_matrix": np.ones((20, 2)),
            "channel_names": ["power", "wind_speed"],
        },
    )
    values = dict(plugin.validate(request))
    execution, trace = plugin.workflow(request).steps[0].execute(values), None

    assert captured["residual_threshold"] == 4.5
    assert captured["cusum_threshold"] == 14.0
    assert execution["wind_calibration"]["development_pool_id"] == "independent-healthy-v1"
