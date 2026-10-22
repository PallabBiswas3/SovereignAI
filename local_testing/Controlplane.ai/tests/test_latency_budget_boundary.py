from pathlib import Path
from itertools import chain, repeat

import pytest

import controlplane.checker as checker_module
from controlplane.checker import ControlPlane
from controlplane.policy import PolicyRepository
from controlplane.schema import EnforcementAction, Interaction


@pytest.mark.parametrize("elapsed_ms,exceeded", [(1999, False), (2000, False), (2001, True)])
def test_internal_two_second_budget_boundary(monkeypatch, elapsed_ms, exceeded):
    policies = Path(__file__).resolve().parents[3] / "Controlplane.ai/configs/policies"
    profile = PolicyRepository(policies).load("internal_assistant")
    assert profile.latency_budget_ms == 2000
    assert profile.version == "1.3"
    # Isolate the actual checker's timing boundary from detector/model latency.
    for check in profile.checks.values():
        check.enabled = False
    # Profiling adds clock reads; all work after the start remains at the
    # prescribed boundary instead of exhausting a fixed number of reads.
    ticks = chain([0.0], repeat(elapsed_ms / 1000.0))
    monkeypatch.setattr(checker_module, "perf_counter", lambda: next(ticks))
    report = ControlPlane(audit_enabled=False).check(
        Interaction(prompt="Timing boundary probe", response="", consequential=True), profile,
    )
    assert report.decision.latency_budget_exceeded is exceeded
    assert report.decision.action == (EnforcementAction.REVIEW if exceeded else EnforcementAction.ALLOW)
