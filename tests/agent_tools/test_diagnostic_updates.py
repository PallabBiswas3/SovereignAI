"""Real adapter/API checks on synthetic fixtures; no accuracy claims."""
from copy import deepcopy
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from diagnostic_fixtures import payload, timestamps
from tsdiag.api import app
from tsdiag.input_contract import catalogue, model_input_profile, validate_payload
from tsdiag.integrations import diagnose_tool
from tsdiag.registry import model_registry
from tsdiag.runtime_updates import (
    BatteryFadeAdapter, BearingCycloAdapter, DiagnosticUpdate,
    TransformerSequenceAdapter, TurbofanRULAdapter, WindEventAdapter,
    WindResidualProvider, register_diagnostic_update,
)


@pytest.fixture(autouse=True)
def isolated_registry():
    before = dict(model_registry._records)
    yield
    model_registry._records.clear()
    model_registry._records.update(before)


def register(data, adapter, **kwargs):
    ref = f"synthetic-{adapter.domain}-{adapter.slot}"
    data["model_refs"] = {adapter.slot: ref}
    return register_diagnostic_update(ref, adapter, version="test-v1",
        input_profile=model_input_profile(data), training_dataset="synthetic-training",
        calibration_dataset="synthetic-calibration", synthetic=True, **kwargs)


def request(data):
    result = TestClient(app).post("/v1/diagnose", json=data)
    assert result.status_code == 200, result.text
    output = result.json()
    json.dumps(output, allow_nan=False)
    return output


def update(output):
    assert output.get("diagnostic_updates"), output.get("abstain_reason")
    report = output["diagnostic_updates"][0]
    DiagnosticUpdate.model_validate(report)
    assert report["model_ref"] in output["metadata"]["model_refs"].values()
    assert report["version"] == "test-v1"
    assert report["synthetic_artifact"]
    assert output["provenance"]["artifact_checksums"]
    assert output["tool_trace"][-1]["tool"] == "diagnostic_update"
    return report


@pytest.fixture(scope="module")
def sequence_adapter():
    from tsdiag.challengers.transformer_sequence import SequenceFeatureClassifier, SequenceFeatureExtractor
    from tsdiag.uncertainty import ConformalClassifier
    rng = np.random.default_rng(41)
    def wave(label):
        phase = np.arange(100)[:, None] * 4 * np.pi / 100 + np.array([0, -2*np.pi/3, 2*np.pi/3])
        return np.hstack([np.cos(phase), (.5 + label*2) * np.cos(phase-.3)]) + rng.normal(0, .01, (100, 6))
    y = np.repeat([0, 1], 40)
    train = np.stack([wave(label) for label in y])
    model = SequenceFeatureClassifier(SequenceFeatureExtractor(2), min_per_class=20).fit(train, y)
    cal_y = np.repeat([0, 1], 25)
    cal_x = np.stack([wave(label) for label in cal_y])
    cc = ConformalClassifier(.1, class_conditional=True).fit(model.predict_proba(cal_x), cal_y)
    return TransformerSequenceAdapter(model, cc, {0: "normal", 1: "winding_looseness"}), wave(1)


def transformer_data(wave):
    data = payload("transformer")
    data["inputs"].update(signal_matrix=wave.tolist(), sampling_rate_hz=2500., fundamental_hz=50.,
        sensor_positions=["Ua", "Ub", "Uc", "Ia", "Ib", "Ic"], channel_units=["V"]*3+["A"]*3)
    return data


def test_sequence_classifier_is_used_by_public_api(sequence_adapter):
    adapter, wave = sequence_adapter
    data = transformer_data(wave)
    register(data, adapter)
    output = request(data)
    report = update(output)
    assert report["calibrated"]
    assert report["prediction_set"] == ["winding_looseness"]
    assert output["metadata"]["model_family"] == adapter.model_family
    assert output["uncertainty_estimate"]["details"]["applies_to"] == "sequence_classifier"


def test_ambiguous_sequence_prediction_holds_even_if_baseline_would_monitor(sequence_adapter):
    adapter, wave = deepcopy(sequence_adapter)
    adapter.conformal.qhat_[:] = np.inf
    data = transformer_data(wave)
    register(data, adapter)
    output = request(data)
    report = update(output)
    assert report["status"] == "abstained"
    assert len(report["prediction_set"]) == 2
    assert output["decision"] == "abstain"
    assert output["hypotheses"] == []
    assert output["confidence"] == 0.


def test_probability_column_indices_map_to_noncontiguous_class_labels(sequence_adapter):
    adapter, wave = deepcopy(sequence_adapter)
    adapter.model.classes_ = np.array([10, 30])
    adapter.labels = {10: "normal", 30: "winding_looseness"}
    assert adapter(wave)["diagnostic_update"]["prediction_set"] == ["winding_looseness"]


def test_wrong_phase_order_is_rejected_before_model_runs(sequence_adapter):
    adapter, wave = sequence_adapter
    data = transformer_data(wave)
    register(data, adapter)
    data["inputs"]["sensor_positions"] = ["Ia", "Ib", "Ic", "Ua", "Ub", "Uc"]
    output = request(data)
    assert output["decision"] == "abstain"
    assert not output["diagnostic_updates"]
    assert "phase order" in output["abstain_reason"]


@pytest.fixture(scope="module")
def rul_adapter():
    from tsdiag.challengers.turbofan_rul import RULChallenger
    rng = np.random.default_rng(62)
    windows = rng.normal(size=(80, 4, 2))
    model = RULChallenger(n_members=1, window=4).fit(windows, np.full(80, 12.), np.repeat(["train-a", "train-b"], 40))
    model.calibrate(windows[:20], np.full(20, 13.), units=[f"cal-{i}" for i in range(20)])
    return TurbofanRULAdapter(model)


def test_calibrated_ensemble_reaches_existing_rul_policy(rul_adapter):
    data = payload("turbofan")
    data["inputs"]["cycle_index"] = list(range(1, 33))
    register(data, rul_adapter)
    output = request(data)
    report = update(output)
    assert report["point_estimate"] == pytest.approx(12.)
    assert output["prognosis"]["remaining_useful_life"] == pytest.approx(12.)
    assert output["prognosis"]["details"]["interval_calibrated"]
    assert output["uncertainty_estimate"]["calibration_dataset"] == "synthetic-calibration"


def test_wide_rul_intervals_do_not_bypass_policy(rul_adapter):
    adapter = deepcopy(rul_adapter)
    adapter.model.conformal_.qhat_ = 10000.
    data = payload("turbofan")
    data["inputs"]["cycle_index"] = list(range(1, 33))
    register(data, adapter)
    output = request(data)
    report = update(output)
    assert output["decision"] == "abstain"
    assert output["abstain_reason"] == "rul_interval_too_wide"
    assert output["prognosis"]["remaining_useful_life"] is None
    assert report["details"]["policy_hold_remains"]


def test_rul_engine_overlap_cannot_cross_public_boundary(rul_adapter):
    data = payload("turbofan")
    data["inputs"]["cycle_index"] = list(range(1, 33))
    data["inputs"]["engine_id"] = data["run_context"]["metadata"]["specimen_id"] = "train-a"
    register(data, rul_adapter)
    output = request(data)
    assert output["decision"] == "abstain"
    assert "disjoint" in output["abstain_reason"]


@pytest.mark.parametrize("kind", ["pf", "gp"])
def test_battery_forecast_is_returned_without_mutating_registered_template(kind):
    from tsdiag.challengers.battery_prognostics import FadeParticleFilter, GPFadeModel, fade_curve
    model = FadeParticleFilter(1.4, n_particles=256, seed=7) if kind == "pf" else GPFadeModel(1.4, n_samples=30, horizon=300, seed=7)
    data = payload("battery_prognosis")
    cycles = np.arange(80, dtype=float)
    data["inputs"].update(cycle_index=cycles.tolist(), capacity_ah=fade_curve(cycles, 2., .0025, .006).tolist())
    record = register(data, BatteryFadeAdapter(model))
    first, second = request(data), request(data)
    report = update(first)
    assert not report["calibrated"]
    assert report["status"] == "completed", report
    assert report["point_estimate"] >= 0
    assert first["prognosis"]["remaining_useful_life"] == second["prognosis"]["remaining_useful_life"]
    assert first["hypotheses"][0]["details"]["rul_cycles"] == report["point_estimate"]
    assert first["prognosis"]["risk"] is None
    assert not record.artifact.model._fitted


def test_bearing_features_do_not_lift_existing_abstention():
    data = payload("bearing")
    register(data, BearingCycloAdapter(healthy_scores=tuple(np.linspace(0, 1, 30))))
    output = request(data)
    report = update(output)
    assert "ies_score_BPFO" in report["details"]["features"]
    assert report["details"]["supplementary_only"]
    assert output["decision"] == "abstain"
    assert report["details"]["policy_hold_remains"]


def test_fitted_snapshot_cannot_be_modified_after_registration():
    data = payload("bearing")
    record = register(data, BearingCycloAdapter())
    record.artifact.alpha = .2
    output = request(data)
    assert output["decision"] == "abstain"
    assert "changed after registration" in output["abstain_reason"]


def test_runtime_specimen_is_checked_against_declared_training_units():
    data = payload("bearing")
    register(data, BearingCycloAdapter(), excluded_specimen_ids=["specimen-1"])
    output = request(data)
    assert output["decision"] == "abstain"
    assert "overlaps" in output["abstain_reason"]


def test_unselected_updates_leave_baseline_execution_unchanged():
    data = payload("bearing")
    reference = diagnose_tool(data)
    register(data, BearingCycloAdapter())
    data["model_refs"] = {}
    actual = diagnose_tool(data)
    assert actual["decision"] == reference["decision"]
    assert actual["detection"] == reference["detection"]
    assert actual["diagnostic_updates"] == []
    assert [s["tool"] for s in actual["tool_trace"]] == [s["tool"] for s in reference["tool_trace"]]


def test_missing_selected_artifact_does_not_silently_use_baseline():
    data = payload("bearing")
    data["model_refs"] = {"diagnostic_update": "not-installed"}
    output = request(data)
    assert output["decision"] == "abstain"
    assert "Unknown model ref" in output["abstain_reason"]


def test_bare_object_cannot_be_selected_as_update():
    data = payload("bearing")
    data["model_refs"] = {"diagnostic_update": "bare-object"}
    model_registry.register("bare-object", version="1", artifact=object(), checksum="test",
        metadata={"input_profile": model_input_profile(data)})
    output = request(data)
    assert output["decision"] == "abstain"
    assert "registered diagnostic update adapter" in output["abstain_reason"]


@pytest.mark.parametrize("bad", ["client_threshold", "raw_model_object", "calibration_probabilities"])
def test_clients_cannot_supply_model_or_calibration_controls(bad):
    data = payload("bearing")
    data["inputs"][bad] = 1.
    assert TestClient(app).post("/v1/diagnose", json=data).status_code == 422


def test_optional_speed_and_setting_vectors_are_shape_checked():
    bearing = payload("bearing")
    bearing["inputs"]["shaft_speed_rpm_trace"] = [1800.]*8
    with pytest.raises(ValueError, match="align"):
        validate_payload(bearing)
    turbo = payload("turbofan")
    turbo["inputs"]["operating_settings"] = [[0., 0., 0.]]*8
    with pytest.raises(ValueError, match="align"):
        validate_payload(turbo)


def test_update_catalogue_matches_model_slots_and_response_schema():
    from tsdiag.integrations.diagnostic_tool import DIAGNOSE_TOOL_MANIFEST
    c = catalogue()
    assert c["diagnostic_update_schema"]["properties"]["status"]["enum"] == ["completed", "abstained"]
    for workflow in c["workflows"]:
        assert set(workflow["model_slot_descriptions"]) == set(workflow["model_slots"])
        variant = next(v["properties"] for v in DIAGNOSE_TOOL_MANIFEST["input_schema"]["oneOf"]
            if v["properties"]["domain"]["const"] == workflow["domain"] and v["properties"]["task"]["const"] == workflow["task"])
        assert set(variant["model_refs"]["properties"]) == set(workflow["model_slots"])


def test_backend_validation_preserves_selected_update_id():
    from app.integrations.models import DiagnosticInvocation
    data = payload("bearing")
    data["model_refs"] = {"diagnostic_update": "bearing-v1"}
    assert DiagnosticInvocation.model_validate(data).model_refs == data["model_refs"]


def test_generated_frontend_contract_is_current():
    import runpy
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    generator = runpy.run_path(str(root/"scripts/generate_diagnostic_update_types.py"))
    assert (root/"frontend/lib/diagnostic-update.ts").read_text() == generator["generated_type"]()


@pytest.fixture(scope="module")
def wind_adapter():
    from tsdiag.tools.wind_scada import fit_wind_normal_behavior_model
    from tsdiag.calibration import EventScorer, EventFARCalibrator, SplitManifest
    rng = np.random.default_rng(78)
    def measurements(n):
        driver = rng.uniform(8, 12, n)
        return np.column_stack([driver, 40 + 2*driver + rng.normal(0, .2, n)])
    train = measurements(400)
    state = fit_wind_normal_behavior_model(train, np.zeros(400, dtype=int), target_indices=[1], predictor_indices=[0])
    provider = WindResidualProvider(state)
    events = {f"cal-{i}": provider({"train_matrix": measurements(200), "prediction_matrix": measurements(80)}) for i in range(30)}
    calibrator = EventFARCalibrator(EventScorer(window=3), .1)
    calibrator.fit(events, SplitManifest("event", tuple(events), ("specimen-1",)).validate(), n_boot=0)
    return WindEventAdapter(calibrator, provider), measurements(200), measurements(80)


def wind_data(train, current):
    data = payload("wind_scada")
    data["inputs"].update(train_matrix=train.tolist(), prediction_matrix=current.tolist(),
        channel_names=["wind_speed", "gearbox_temp"], channel_units=["m/s", "degC"],
        train_timestamps=timestamps(len(train)), prediction_timestamps=timestamps(len(current), 300))
    data["run_context"]["metadata"].update(observation_timestamp_utc=timestamps(1, 300)[0], acquisition_end_utc=timestamps(1, 400)[0])
    return data


def test_wind_calibrator_uses_frozen_nbm_and_is_supplementary(wind_adapter):
    adapter, train, current = wind_adapter
    data = wind_data(train, current)
    baseline = diagnose_tool(data)
    register(data, adapter)
    output = request(data)
    report = update(output)
    assert report["method"] == "wind-event-conformal-v1"
    assert report["calibrated"]
    assert isinstance(report["details"]["alarm"], bool)
    assert report["details"]["calibration_sha256"]
    assert output["decision"] == baseline["decision"]


def test_unscorable_wind_has_no_nan_or_false_healthy_result(wind_adapter):
    from tsdiag.calibration import EventScorer, EventFARCalibrator, SplitManifest
    original, train, current = wind_adapter
    adapter = deepcopy(original)
    rng = np.random.default_rng(923)
    events = {}
    for i in range(12):
        x = np.tile(train, (2, 1)) + rng.normal(0, .1, (400, 2))
        events[f"long-{i}"] = adapter.residual_provider({"train_matrix": train, "prediction_matrix": x})
    adapter.calibrator = EventFARCalibrator(EventScorer(window=128), .1)
    adapter.calibrator.fit(events, SplitManifest("event", tuple(events), ("specimen-1",)).validate(), n_boot=0)
    data = wind_data(train, current)
    register(data, adapter)
    output = request(data)
    report = update(output)
    assert report["status"] == "abstained"
    assert report["details"]["alarm"] is None
    assert report["details"]["score"] is None
    json.dumps(output, allow_nan=False)


def test_wind_rejects_reused_calibration_event_identity(wind_adapter):
    adapter, train, current = wind_adapter
    data = wind_data(train, current)
    data["run_context"]["metadata"]["specimen_id"] = "cal-0"
    register(data, adapter)
    output = request(data)
    assert output["decision"] == "abstain"
    assert "disjoint" in output["abstain_reason"]


def test_registered_model_cannot_be_used_in_another_slot(sequence_adapter):
    adapter, wave = sequence_adapter
    data = transformer_data(wave)
    record = register(data, adapter)
    data["model_refs"] = {"trained_image_model": record.ref}
    output = request(data)
    assert output["decision"] == "abstain"
    assert "domain/task/slot" in output["abstain_reason"]


def test_censored_battery_forecast_withholds_baseline_point():
    from tsdiag.challengers.battery_prognostics import GPFadeModel, fade_curve
    data = payload("battery_prognosis")
    cycles = np.arange(80, dtype=float)
    data["inputs"].update(cycle_index=cycles.tolist(), capacity_ah=fade_curve(cycles, 2., .0025, .006).tolist())
    register(data, BatteryFadeAdapter(GPFadeModel(1.4, horizon=1, n_samples=30, seed=9)))
    output = request(data)
    report = update(output)
    assert report["status"] == "abstained"
    assert report["point_estimate"] is None and report["interval"] is None
    assert output["decision"] == "abstain"
    assert output["prognosis"]["remaining_useful_life"] is None
    assert output["uncertainty_estimate"]["value"] is None


def test_synthesis_preserves_update_caveats_and_keeps_unlinked_data_out():
    from app.integrations.orchestrator import IndustrialIntegrationOrchestrator
    diagnostic = {"decision": "abstain", "diagnostic_updates": [{"method": "sequence", "status": "abstained", "synthetic_artifact": True, "prediction_set": ["normal", "fault"]}]}
    text = IndustrialIntegrationOrchestrator._candidate_prompt("Assess the asset", graph=None, diagnostic=diagnostic, diagnostic_linked=True)
    assert '"status": "abstained"' in text
    assert "Supplementary alarms are not confirmed faults" in text
    assert "synthetic artifacts" in text
    unlinked = IndustrialIntegrationOrchestrator._candidate_prompt("Assess the asset", graph=None, diagnostic=diagnostic, diagnostic_linked=False)
    assert "SELECTED DIAGNOSTIC ANALYSIS" not in unlinked


def test_wrong_waveform_cycle_metadata_cannot_reuse_fixed_sequence_extractor(sequence_adapter):
    adapter, wave = sequence_adapter
    data = transformer_data(wave)
    data["inputs"]["signal_matrix"] = np.tile(wave, (2, 1)).tolist()
    register(data, adapter)
    output = request(data)
    assert output["decision"] == "abstain"
    assert "sequence cycles" in output["abstain_reason"]


def test_rul_requires_settings_when_regime_preprocessing_is_registered(rul_adapter):
    from tsdiag.challengers.turbofan_rul import RegimeNormalizer
    adapter = deepcopy(rul_adapter)
    rng = np.random.default_rng(149)
    adapter.normalizer = RegimeNormalizer(1).fit(rng.normal(size=(100, 3)), rng.normal(size=(100, 2)))
    data = payload("turbofan")
    data["inputs"]["cycle_index"] = list(range(1, 33))
    register(data, adapter)
    missing = request(data)
    assert missing["decision"] == "abstain"
    assert "operating_settings" in missing["abstain_reason"]
    data["inputs"]["operating_settings"] = rng.normal(size=(32, 3)).tolist()
    output = request(data)
    assert update(output)["calibrated"]


def test_nonfinite_or_reversed_public_intervals_are_rejected():
    for interval in [(3., 1.), (0., float("inf")), (-1., 4.)]:
        with pytest.raises(ValueError):
            DiagnosticUpdate(method="test", summary="bad", interval=interval)


def test_application_http_client_preserves_actual_service_update_response():
    import asyncio
    import httpx
    from app.integrations.clients import DiagnosticAgentClient
    data = payload("bearing")
    register(data, BearingCycloAdapter())

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8200") as http:
            return await DiagnosticAgentClient("http://127.0.0.1:8200", client=http, max_retries=0).diagnose(data)

    report = update(asyncio.run(call()))
    assert report["method"] == "bearing-modulation-order-v1"


def test_measured_speed_trace_connects_order_tracking():
    data = payload("bearing")
    data["inputs"]["signal"] = data["inputs"]["signal"] * 8
    data["inputs"]["shaft_speed_rpm_trace"] = [1800.]*len(data["inputs"]["signal"])
    register(data, BearingCycloAdapter(band=(1500., 2500.)))
    report = update(request(data))
    assert np.isfinite(report["details"]["dominant_order"])


def test_order_tracking_withholds_unresolvable_short_record():
    data = payload("bearing")
    data["inputs"]["shaft_speed_rpm_trace"] = [1800.]*len(data["inputs"]["signal"])
    register(data, BearingCycloAdapter(band=(1500., 2500.)))
    output = request(data)
    assert output["decision"] == "abstain"
    assert "two shaft revolutions" in output["abstain_reason"]
