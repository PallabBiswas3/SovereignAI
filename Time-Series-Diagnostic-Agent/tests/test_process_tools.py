import numpy as np

from tsdiag.domains import ProcessDiagnosticPipeline
from tsdiag.evaluation import evaluate_process_predictions
from tsdiag.tools import (
    causal_graph_filter,
    fault_onset_timing,
    process_diagnosis,
    root_cause_rank,
)


def test_causal_graph_filter_respects_topology():
    edges = [
        {"cause": "cooling_flow", "effect": "temperature", "p_value": 0.001},
        {"cause": "pressure", "effect": "temperature", "p_value": 0.002},
    ]
    topology = {"cooling_flow": ["temperature"]}
    result = causal_graph_filter(edges, topology)
    kept = result["filtered_causal_graph"]["edges"]
    assert len(kept) == 1
    assert kept[0]["cause"] == "cooling_flow"
    assert len(result["removed_edges"]) == 1


def test_fault_onset_timing_orders_channels():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(0, 0.2, size=(n, 3))
    x[120:, 0] += 3.0
    x[150:, 1] += 2.5
    alarm = np.arange(n) >= 115
    result = fault_onset_timing(
        x,
        alarm,
        ["root", "downstream", "stable"],
        z_threshold=4.0,
        persistence=3,
    )
    assert result["onset_order"][0] == "root"
    assert result["onset_times"]["root"] < result["onset_times"]["downstream"]


def test_root_cause_rank_prefers_early_upstream_variable():
    graph = {
        "edges": [
            {"cause": "A", "effect": "B"},
            {"cause": "A", "effect": "C"},
            {"cause": "B", "effect": "C"},
        ]
    }
    result = root_cause_rank(
        [0, 1],
        graph,
        ["A", "B", "C"],
        variable_contributions=np.array([0.5, 0.3, 0.2]),
        channel_names=["A", "B", "C"],
    )
    assert result["root_cause_ranking"][0]["variable"] == "A"
    assert ["A", "B"] in result["propagation_paths"]


def test_process_diagnosis_uses_fault_catalog():
    ranking = [{"variable": "cooling_flow", "score": 0.8}]
    paths = [["cooling_flow", "temperature"], ["cooling_flow", "temperature", "pressure"]]
    catalog = {
        "faults": [
            {
                "label": "cooling-water disturbance",
                "root_cause": "cooling_flow",
                "affected_variables": ["temperature", "pressure"],
            }
        ]
    }
    result = process_diagnosis(ranking, paths, catalog)
    assert result["fault_label"] == "cooling-water disturbance"
    assert result["root_cause"] == "cooling_flow"


def test_process_pipeline_detects_synthetic_fault():
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(1200, 3))
    cur = rng.normal(size=(500, 3))
    # Fault starts in A and propagates into B/C with delays.
    cur[180:, 0] += 5.0
    cur[220:, 1] += 3.0
    cur[260:, 2] += 2.0

    pipeline = ProcessDiagnosticPipeline(maxlag=1, onset_z_threshold=3.0, diagnosis_threshold=0.2)
    result = pipeline.run(
        cur,
        ref,
        ["A", "B", "C"],
        process_topology={"A": ["B", "C"], "B": ["C"]},
        fault_catalog={"A": "synthetic upstream fault"},
    )
    assert result.fault_detected
    assert result.root_cause in {"A", "B", "C"}
    assert "dpca_monitoring" in result.tool_trace
    assert "cva_monitoring" in result.tool_trace
    assert "hybrid_detection_arbitration" in result.tool_trace
    assert "process_diagnosis" in result.tool_trace


def test_process_evaluation_metrics():
    metrics = evaluate_process_predictions([
        {
            "true_fault_detected": False,
            "predicted_fault_detected": False,
        },
        {
            "true_fault_detected": True,
            "predicted_fault_detected": True,
            "true_fault_label": "F1",
            "predicted_fault_label": "F1",
            "true_root_cause": "A",
            "predicted_root_cause": "A",
            "true_onset_index": 100,
            "predicted_onset_index": 108,
        },
    ])
    assert metrics["detection_f1"] == 1.0
    assert metrics["root_cause_accuracy"] == 1.0
    assert metrics["detection_delay"] == 8.0


def test_process_hybrid_dpca_only_alarm_is_early_warning(monkeypatch):
    import tsdiag.domains.process_runner as runner_module

    def fake_monitor(signal, reference, config):
        mask = np.zeros(len(signal), dtype=bool)
        mask[:10 if config.method == "dpca" else 1] = True
        result = {"alarm_mask": mask}
        if config.method == "cva":
            result["variable_contributions"] = np.ones_like(signal, dtype=float)
        return result

    monkeypatch.setattr(runner_module, "run_monitoring_method", fake_monitor)
    ref = np.zeros((100, 3))
    result = ProcessDiagnosticPipeline(minimum_alarm_fraction=0.05).run(
        ref.copy(), ref, ["A", "B", "C"]
    )
    assert not result.fault_detected
    assert result.artifacts["detection"]["early_warning"]
    assert result.artifacts["detection"]["status"] == "early_warning"
