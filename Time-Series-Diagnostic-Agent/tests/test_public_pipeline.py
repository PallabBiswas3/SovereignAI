import numpy as np
import json

from tsdiag import PIPELINE_VERSION, DiagnosticPipeline, DiagnosticResult, diagnose, get_domain_pack
from tsdiag.domains import default_domain_tool_registry
from tsdiag.execution import ExecutionTrace


def _trace_names(result):
    trace = result.tool_trace
    if isinstance(trace, ExecutionTrace):
        return [step.tool for step in trace.flatten()]
    rows = []
    def visit(step):
        rows.append(step.tool)
        for child in getattr(step, "children", []): visit(child)
    for step in trace: visit(step)
    return rows


def test_missing_required_metadata_returns_structured_abstention():
    result = diagnose("bearing", signal=np.ones(128))
    assert isinstance(result, DiagnosticResult)
    assert result.decision == "abstain"
    assert "sampling_rate_hz" in result.abstain_reason


def test_bearing_runs_through_public_pipeline():
    fs = 8000.0
    t = np.arange(0, 1.5, 1 / fs)
    x = (1 + 0.8 * np.sin(2 * np.pi * 80 * t)) * np.sin(2 * np.pi * 1800 * t)
    result = diagnose("bearing", signal=x, sampling_rate_hz=fs,
                      fault_frequencies={"BPFO": 80.0, "BPFI": 125.0, "BSF": 55.0, "FTF": 10.0})
    assert result.domain == "bearing"
    assert result.decision in {"diagnose", "abstain"}
    assert result.tool_trace
    assert all(step.duration_seconds is not None for step in result.tool_trace)
    assert result.tool_trace[-1].tool == "bearing_evidence_fusion"


def test_process_uses_decomposed_cross_domain_workflow():
    rng = np.random.default_rng(4)
    ref = rng.normal(size=(180, 3))
    cur = rng.normal(size=(100, 3))
    result = diagnose("process", signal_matrix=cur, normal_reference=ref,
                      channel_names=["pressure", "flow", "level"], sampling_rate_hz=1.0, maxlag=1)
    assert isinstance(result, DiagnosticResult)
    assert result.domain == "process"
    assert result.metadata["pipeline_version"] == PIPELINE_VERSION
    assert all(step.duration_seconds is not None for step in result.tool_trace)
    assert result.tool_trace[0].tool == "standardize_against_normal"
    assert result.tool_trace[-1].tool == "process_diagnosis"
    assert result.metadata["workflow_version"] == "3.0"
    assert result.metadata["policy_version"] == "process-policy-v3"


def test_wind_scada_pipeline_localizes_persistent_shift():
    rng = np.random.default_rng(5)
    ref = rng.normal(scale=0.2, size=(200, 3))
    cur = rng.normal(scale=0.2, size=(200, 3))
    cur[80:, 1] += 2.0
    result = diagnose(
        "wind_scada",
        train_matrix=ref,
        prediction_matrix=cur,
        channel_names=["power", "gearbox_temp", "wind"],
        prediction_timestamps=np.arange(200),
        event_ood_abstain_fraction=1.0,
    )
    assert result.decision == "diagnose"
    assert "gearbox_temp" in result.localization.channels
    assert [step.tool for step in result.tool_trace] == ["wind_analysis", "event_decision"]
    assert result.metadata["policy_version"] == "wind-care-event-policy:1.0"


def test_battery_pipeline_localizes_outlying_cell():
    rng = np.random.default_rng(6)
    voltage = 3.7 + rng.normal(scale=0.003, size=(80, 5))
    temperature = 30 + rng.normal(scale=0.1, size=(80, 5))
    voltage[:, 3] -= 0.12
    result = diagnose("battery", cell_voltage=voltage, cell_temperature=temperature,
                      cell_ids=[f"cell_{i}" for i in range(5)], timestamps=np.arange(80), cell_anomaly_threshold=1.5)
    assert result.decision == "diagnose"
    assert result.localization.components == ["cell_3"]
    assert result.prognosis is not None
    assert [step.tool for step in result.tool_trace] == list(get_domain_pack("battery").tool_names())
    assert result.metadata["workflow_version"] == "2.0"
    assert result.metadata["policy_version"] == "battery-pack-policy-v2"


def test_turbofan_pipeline_returns_health_and_prognosis():
    rng = np.random.default_rng(7)
    cycles = np.arange(1, 121)
    signal = np.column_stack([0.02 * cycles + rng.normal(scale=.05, size=120), rng.normal(size=120)])
    result = diagnose("turbofan", signal_matrix=signal, channel_names=["temperature", "noise"], cycle_index=cycles)
    assert result.decision in {"diagnose", "monitor"}
    assert result.prognosis is not None
    assert result.localization.channels
    assert [step.tool for step in result.tool_trace] == list(get_domain_pack("turbofan").tool_names())
    assert result.metadata["workflow_version"] == "2.0"
    assert result.metadata["policy_version"] == "turbofan-policy-v2"


def test_transformer_pipeline_uses_decomposed_workflow():
    fs = 4000.0
    t = np.arange(0, 1, 1/fs)
    base = np.sin(2*np.pi*300*t)
    impulses = np.zeros_like(t)
    impulses[::100] = 8
    signal = np.column_stack([base + impulses, .8*base + impulses])
    result = diagnose("transformer", signal_matrix=signal, sampling_rate_hz=fs, sensor_positions=["tank_a", "tank_b"])
    assert result.domain == "transformer"
    assert result.decision in {"monitor", "abstain"}
    assert [step.tool for step in result.tool_trace] == list(get_domain_pack("transformer").tool_names())
    assert result.tool_trace[-1].tool == "transformer_decision"
    assert result.metadata["workflow_version"] == "2.0"
    assert result.metadata["policy_version"] == "transformer-policy-v2"


def test_transformer_prefers_registered_default_ad_tfm_for_sgah_input(monkeypatch):
    from tsdiag.contracts import DiagnosticRequest
    from tsdiag.pipeline import DEFAULT_TRANSFORMER_ADTFM_REF
    from tsdiag.registry import ModelRecord, model_registry

    class DummyADTFM:
        model_family = "AD-TFM-AT"

        def __call__(self, waveform):
            assert np.asarray(waveform).shape == (100, 6)
            return {
                "label": "main_transformer_fault",
                "predicted_class": "main_transformer_fault",
                "confidence": 0.99,
                "probabilities": {"main_transformer_fault": 0.99, "normal": 0.01},
                "model_family": self.model_family,
            }

    monkeypatch.setitem(
        model_registry._records,
        DEFAULT_TRANSFORMER_ADTFM_REF,
        ModelRecord(
            ref=DEFAULT_TRANSFORMER_ADTFM_REF,
            version="ad-tfm-at-test",
            artifact=DummyADTFM(),
            checksum="test-checksum",
        ),
    )
    signal = np.random.default_rng(41).normal(size=(100, 6))
    result = diagnose(DiagnosticRequest(
        domain="transformer",
        task="fault_diagnosis",
        inputs={
            "signal_matrix": signal,
            "sampling_rate_hz": 1.0,
            "sensor_positions": ["Ua", "Ub", "Uc", "Ia", "Ib", "Ic"],
            "anomaly_threshold": 0.0,
        },
    ))

    assert result.decision == "diagnose"
    assert result.metadata["model_family"] == "AD-TFM-AT"
    assert result.metadata["model_refs"]["raw_waveform_model"] == DEFAULT_TRANSFORMER_ADTFM_REF
    assert result.provenance.model_versions["raw_waveform_model"] == "ad-tfm-at-test"


def test_dispatcher_exposes_current_version():
    assert DiagnosticPipeline.version == PIPELINE_VERSION == "1.1.0"


def test_unsupported_domain_task_returns_structured_abstention():
    result = diagnose("battery", task="remaining_useful_life", cell_ids=["c1"], timestamps=np.arange(10))
    assert result.decision == "abstain"
    assert "Unsupported task" in result.abstain_reason


def test_every_domain_contract_has_a_registered_callable():
    registry = default_domain_tool_registry()
    for domain in ("bearing", "process", "wind_scada", "battery", "turbofan", "transformer"):
        assert set(get_domain_pack(domain).tool_names()) <= set(registry.names(domain))


def test_new_domain_trace_steps_are_timed_and_json_safe():
    rng = np.random.default_rng(30)
    result = diagnose("battery", cell_voltage=3.7+rng.normal(scale=.003,size=(40,4)),
                      cell_temperature=30+rng.normal(scale=.1,size=(40,4)),
                      cell_ids=["a","b","c","d"], timestamps=np.arange(40))
    assert all(step.duration_seconds is not None and step.duration_seconds >= 0 for step in result.tool_trace)
    assert [step.tool for step in result.tool_trace] == list(get_domain_pack("battery").tool_names())
    json.loads(result.to_json())


def test_step_failure_is_isolated_with_failed_step_trace(monkeypatch):
    import tsdiag.domains.wind_scada_plugin as plugin_module

    class BrokenPipeline:
        def __init__(self, **kwargs): pass
        def run(self, *args, **kwargs): raise RuntimeError("model unavailable")

    monkeypatch.setattr(plugin_module, "WindScadaDiagnosticPipeline", BrokenPipeline)
    result = diagnose(
        "wind_scada",
        train_matrix=np.ones((40, 2)),
        prediction_matrix=np.ones((40, 2)),
        channel_names=["a", "b"],
    )
    assert result.decision == "abstain"
    assert result.tool_trace[-1].tool == "wind_analysis"
    assert result.tool_trace[-1].status == "error"
    assert result.tool_trace[-1].duration_seconds is not None
    assert "model unavailable" in result.abstain_reason
