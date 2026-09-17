from types import SimpleNamespace

import numpy as np

from tsdiag.contracts import DiagnosticRequest, RunContext
from tsdiag.execution import ExecutionTrace, Step, Workflow, WorkflowExecutor
from tsdiag.models import ToolTraceStep
from tsdiag.pipeline import diagnose
from tsdiag.registry import domain_registry, model_registry, policy_registry


def test_execution_trace_preserves_order_and_named_lookup():
    trace = ExecutionTrace([
        ToolTraceStep("analysis", status="ok", children=[ToolTraceStep("inner", status="ok")]),
        ToolTraceStep("verification", status="warning"),
        ToolTraceStep("analysis", status="ok"),
    ])
    assert [step.tool for step in trace.steps] == ["analysis", "verification", "analysis"]
    assert trace.get("analysis") is trace.steps[-1]
    assert trace.get("analysis", occurrence=0) is trace.steps[0]
    assert trace.require("verification").status == "warning"
    assert trace.require("inner").tool == "inner"
    assert [step.tool for step in trace.flatten()] == ["analysis", "inner", "verification", "analysis"]


def test_workflow_executor_respects_dependencies_and_named_trace():
    workflow = Workflow(steps=(
        Step("first", lambda state: {"x": 2}),
        Step("second", lambda state: {"y": state["x"] + 3}, depends_on=("first",)),
    ))
    state, trace = WorkflowExecutor().run(workflow, {})
    assert state["y"] == 5
    assert trace.require("first").status == "ok"
    assert trace.require("second").outputs_summary["output_keys"] == ["y"]


def test_wind_pipeline_exports_share_one_implementation():
    from tsdiag.domains import WindScadaBenchmarkPipeline, WindScadaDiagnosticPipeline
    from tsdiag.domains.runners import WindScadaDiagnosticPipeline as RunnerExport

    assert WindScadaDiagnosticPipeline is WindScadaBenchmarkPipeline is RunnerExport


def test_structured_wind_request_uses_plugin_policy(monkeypatch):
    import tsdiag.domains.wind_scada_plugin as plugin_module

    class FakePipeline:
        init_kwargs = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.init_kwargs.append(kwargs)

        def run(self, train_matrix, prediction_matrix, channel_names, **kwargs):
            n = len(prediction_matrix)
            residual = np.zeros(n, dtype=bool); drift = np.zeros(n, dtype=bool)
            residual[2:8] = True; drift[2:8] = True
            return SimpleNamespace(
                affected_channels=[channel_names[-1]], confidence=0.8,
                alarm_mask=residual | drift, anomaly_scores=np.ones(n),
                tool_trace=["legacy-internal-step"],
                artifacts={
                    "anomaly_detection": {"alarm_mask": residual},
                    "residual_changepoint": {"alarm_mask": drift, "change_points": [2]},
                    "regime_assignment": {"out_of_distribution_mask": np.zeros(n, dtype=bool)},
                    "physics_consistency": {"verification_findings": [{"channel": channel_names[-1]}]},
                    "fusion": {"fused_alarm_fraction": float(np.mean(residual | drift))},
                },
            )

    monkeypatch.setattr(plugin_module, "WindScadaDiagnosticPipeline", FakePipeline)
    request_inputs = {
        "train_matrix": np.ones((80, 3)), "prediction_matrix": np.ones((20, 3)),
        "channel_names": ["wind_speed", "power", "temperature"],
        "evaluable_mask": np.ones(20, dtype=bool), "event_minimum_corroborated_run": 6,
    }
    result = diagnose(DiagnosticRequest(
        domain="wind_scada", task="condition_monitoring",
        inputs=request_inputs,
        policy_ref="wind-care-event-policy:1.0",
        model_refs={"normal_behavior": "wind-nbm-regime-v1"},
    ))
    assert result.decision == "diagnose"
    assert result.detection.abnormal is True
    assert result.uncertainty is None
    assert result.uncertainty_estimate.method == "not_calibrated"
    assert result.metadata["workflow_version"] == "1.1"
    assert result.metadata["policy_version"] == "wind-care-event-policy:1.0"
    assert result.metadata["policy_ref"] == "wind-care-event-policy:1.0"
    assert result.metadata["resolved_model_versions"] == {"normal_behavior": "wind-nbm-regime-v1"}
    assert result.metadata["model_version"] == "wind-nbm-regime-v1"
    assert result.tool_trace.require("event_decision").evidence_ids == ["wind-event-evidence"]
    assert FakePipeline.init_kwargs[-1]["cusum_drift"] == 0.5
    assert FakePipeline.init_kwargs[-1]["cusum_threshold"] == 10.0
    assert FakePipeline.init_kwargs[-1]["cusum_hold_samples"] == 6

    alias_result = diagnose(
        "wind_scada",
        normal_reference=request_inputs["train_matrix"],
        signal_matrix=request_inputs["prediction_matrix"],
        channel_names=request_inputs["channel_names"],
        timestamps=np.arange(20),
        evaluable_mask=request_inputs["evaluable_mask"],
        event_minimum_corroborated_run=6,
    )
    assert alias_result.decision == result.decision
    assert [step.tool for step in alias_result.tool_trace] == ["wind_analysis", "event_decision"]
    assert alias_result.metadata["workflow_version"] == result.metadata["workflow_version"]
    assert alias_result.metadata["policy_version"] == result.metadata["policy_version"]


def test_all_six_domain_plugins_are_registered_after_diagnose():
    diagnose(DiagnosticRequest(domain="bearing", task="fault_diagnosis", inputs={}))
    assert domain_registry.names() == ("battery", "bearing", "process", "transformer", "turbofan", "wind_scada")
    for domain in domain_registry.names(): assert "default" in policy_registry.refs(domain)


def test_benchmark_policy_and_model_refs_are_registered():
    diagnose(DiagnosticRequest(domain="bearing", task="fault_diagnosis", inputs={}))
    expected_policies = {
        "bearing": "bearing-policy-v2",
        "battery": "capacity-prognosis-policy-v1",
        "turbofan": "turbofan-policy-v2",
        "wind_scada": "wind-care-event-policy:1.0",
    }
    for domain, ref in expected_policies.items():
        assert policy_registry.resolve(domain, ref).ref == ref
    assert model_registry.resolve("wind-nbm-regime-v1").version == "wind-nbm-regime-v1"


def test_versioned_policy_ref_model_provenance_and_decomposed_trace():
    rng = np.random.default_rng(91)
    voltage = 3.7 + rng.normal(scale=0.003, size=(40, 4))
    temperature = 30 + rng.normal(scale=0.1, size=(40, 4))
    model_registry.register("battery-health-test", version="test-v1", metadata={"purpose": "unit-test provenance only"}, replace=True)

    result = diagnose(DiagnosticRequest(
        domain="battery", task="anomaly_localization", policy_ref="battery-pack-policy-v2",
        model_refs={"health_model": "battery-health-test"},
        run_context=RunContext(run_id="fixture-run", source="unit-test", dataset_id="fixture-data", protocol_id="fixture-protocol"),
        inputs={"cell_voltage": voltage, "cell_temperature": temperature, "cell_ids": ["a", "b", "c", "d"], "timestamps": np.arange(40)},
    ))
    assert result.decision in {"diagnose", "monitor"}
    assert result.metadata["workflow_version"] == "2.0"
    assert result.metadata["policy_version"] == "battery-pack-policy-v2"
    assert result.metadata["policy_ref"] == "battery-pack-policy-v2"
    assert result.metadata["resolved_model_versions"] == {"health_model": "test-v1"}
    assert result.tool_trace.require("battery_data_quality", recursive=False)
    assert result.tool_trace.require("battery_decision", recursive=False)
    assert result.provenance is not None
    assert result.provenance.dataset_id == "fixture-data"
    assert result.provenance.protocol_id == "fixture-protocol"
    assert result.provenance.run_id == "fixture-run"
    assert result.provenance.workflow_version == "2.0"
    assert result.provenance.policy_version == "battery-pack-policy-v2"
    assert len(result.provenance.input_hash) == 64
    assert result.uncertainty_estimate is not None
    assert result.abstention is not None


def test_structured_battery_prognosis_uses_task_specific_policy():
    cycles = np.arange(1, 61, dtype=float); capacity = 2.0 - 0.006 * cycles
    result = diagnose(DiagnosticRequest(
        domain="battery", task="prognosis", policy_ref="capacity-prognosis-policy-v1",
        inputs={"cycle_index": cycles, "capacity_ah": capacity, "battery_id": "fixture", "eol_capacity_ah": 1.4},
    ))
    assert result.task == "prognosis"
    assert result.prognosis is not None
    assert result.metadata["workflow_version"] == "prognosis-1.0"
    assert result.metadata["policy_version"] == "capacity-prognosis-policy-v1"
    assert result.provenance is not None
    assert result.provenance.workflow_version == "prognosis-1.0"
    outer = result.tool_trace.require("battery_prognosis", recursive=False)
    assert outer.children


def test_explicit_policy_ref_cannot_silently_select_task_policy():
    cycles = np.arange(1, 61, dtype=float); capacity = 2.0 - 0.006 * cycles
    result = diagnose(DiagnosticRequest(
        domain="battery", task="prognosis", policy_ref="battery-pack-policy-v2",
        inputs={"cycle_index": cycles, "capacity_ah": capacity},
    ))
    assert result.decision == "abstain"
    assert "does not support task" in result.abstain_reason


def test_bearing_is_decomposed_into_named_workflow_steps():
    fs = 8000.0
    t = np.arange(0, 1.0, 1 / fs)
    x = (1 + .8*np.sin(2*np.pi*80*t))*np.sin(2*np.pi*1800*t)
    result = diagnose(DiagnosticRequest(
        domain="bearing", task="fault_diagnosis", policy_ref="bearing-policy-v2",
        inputs={"signal": x, "sampling_rate_hz": fs, "fault_frequencies": {"BPFO":80., "BPFI":125., "BSF":55., "FTF":10.}},
    ))
    assert [step.tool for step in result.tool_trace] == [
        "signal_integrity", "time_domain_features", "welch_psd", "spectral_kurtosis",
        "bandpass_filter", "hilbert_envelope", "envelope_spectrum",
        "bearing_frequency_match", "bearing_evidence_fusion",
    ]
    assert result.metadata["workflow_version"] == "2.0"
    assert result.metadata["policy_version"] == "bearing-policy-v2"
