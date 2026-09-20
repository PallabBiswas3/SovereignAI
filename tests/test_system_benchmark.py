from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_system.py"
spec = importlib.util.spec_from_file_location("benchmark_system", SCRIPT)
assert spec and spec.loader
benchmark_system = importlib.util.module_from_spec(spec)
sys.modules["benchmark_system"] = benchmark_system
spec.loader.exec_module(benchmark_system)


def test_percentile_interpolates_small_samples() -> None:
    assert benchmark_system.percentile([1.0, 2.0], 0.50) == 1.5
    assert benchmark_system.percentile([1.0, 2.0], 0.95) == 1.95
    assert benchmark_system.percentile([], 0.50) is None


def test_summary_reports_latency_ttft_and_success_rate() -> None:
    rows = [
        benchmark_system.Measurement("general", 1, True, 200, 1.0, ttft_seconds=0.2, tokens_per_second=10.0),
        benchmark_system.Measurement("general", 2, True, 200, 2.0, ttft_seconds=0.4, tokens_per_second=12.0),
        benchmark_system.Measurement("general", 3, False, 503, 0.5, error="unavailable"),
    ]
    summary = benchmark_system.summarize_measurements(rows)["general"]
    assert summary["runs"] == 3
    assert summary["successes"] == 2
    assert summary["success_rate"] == 0.6667
    assert summary["total_p50_s"] == 1.5
    assert summary["ttft_p50_s"] == 0.3
    assert summary["tokens_per_second_mean"] == 11.0


def test_process_fixture_is_deterministic_and_has_fault_shift() -> None:
    first = benchmark_system.process_diagnostic_payload(seed=20)
    second = benchmark_system.process_diagnostic_payload(seed=20)
    assert first == second
    inputs = first["inputs"]
    assert len(inputs["normal_reference"]) == 180
    assert len(inputs["signal_matrix"]) == 100
    before = sum(row[0] for row in inputs["signal_matrix"][:40]) / 40
    after = sum(row[0] for row in inputs["signal_matrix"][40:]) / 60
    assert after - before > 3.0


def test_graph_and_full_payloads_match_integration_contract() -> None:
    graph = benchmark_system.graph_integration_payload()
    full = benchmark_system.full_integration_payload()
    assert graph["include_graph_evidence"] is True
    assert graph["diagnostic"] is None
    assert full["include_graph_evidence"] is True
    assert full["diagnostic"]["domain"] == "process"
    assert full["assurance_level"] == "thorough"
