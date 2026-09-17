import numpy as np
import pytest

from tsdiag import SignalRecord
from tsdiag.agents import (
    BearingDiagnosticAgent,
    CausalRootCauseAgent,
    IndustrialDiagnosticOrchestrator,
    SignalProcessingAgent,
)


def test_signal_processing_agent_runs():
    fs = 2000.0
    t = np.arange(0, 2.0, 1/fs)
    x = np.sin(2*np.pi*120*t)
    result = SignalProcessingAgent().run(SignalRecord(x, fs))
    assert result.agent == "signal_processing"
    assert result.status in {"ok", "warning"}
    assert "dominant_hz" in result.metrics


def test_bearing_agent_abstains_without_fault_frequencies():
    fs = 2000.0
    x = np.random.default_rng(0).normal(size=4000)
    result = BearingDiagnosticAgent().run(SignalRecord(x, fs))
    assert result.status == "abstain"


def test_causal_agent_needs_multichannel_signal():
    fs = 100.0
    x = np.random.default_rng(0).normal(size=1000)
    result = CausalRootCauseAgent().run(SignalRecord(x, fs))
    assert result.status == "abstain"


def test_orchestrator_routes_reference_agents():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(800, 2))
    cur = rng.normal(size=(500, 2))
    record = SignalRecord(cur, 100.0, channels=["pressure", "flow"])
    with pytest.warns(DeprecationWarning, match="deprecated v0 API"):
        report = IndustrialDiagnosticOrchestrator().run(
            record,
            {"reference_windows": ref, "maxlag": 1},
        )
    names = {r.agent for r in report.agent_results}
    assert "signal_processing" in names
    assert "statistical_monitoring" in names
    assert "probabilistic" in names
    assert "transfer_robustness" in names
    assert "causal_root_cause" in names
