import asyncio
from types import SimpleNamespace

import pytest

from app.integrations.clients import IntegrationServiceError, validate_internal_service_url
from app.integrations.models import IntegratedAnalysisRequest
from app.integrations.orchestrator import IndustrialIntegrationOrchestrator


class FakeGraph:
    def __init__(self, *, fail: bool = False):
        self.called = False
        self.fail = fail
        self.verification_mode = None

    async def health(self):
        return {"status": "ok"}

    async def retrieve(self, query, verification_mode="thorough"):
        self.called = True
        self.verification_mode = verification_mode
        if self.fail:
            raise IntegrationServiceError("graph-rag", "offline")
        return {
            "status": "grounded",
            "claims": [{"id": "claim-1", "claim_text": "Pump vibration exceeds the approved limit."}],
            "chunks": [{"id": "chunk-1", "content": "Approved vibration limit: 7.1 mm/s."}],
        }


class FakeDiagnostics:
    def __init__(self):
        self.payload = None

    async def health(self):
        return {"status": "ok"}

    async def diagnose(self, payload):
        self.payload = payload
        return {
            "decision": "fault_detected",
            "detection": {"abnormal": True},
            "hypotheses": [{"label": "bearing wear"}],
            "recommended_actions": ["Inspect the bearing"],
        }


class FakeControlPlane:
    def __init__(self, *, precheck_action="allow", release_action="allow"):
        self.precheck_action = precheck_action
        self.release_action = release_action
        self.checked = None

    async def health(self):
        return {"status": "ok"}

    async def precheck(self, payload):
        return {
            "decision": {"action": self.precheck_action},
            "final_response": "Request held by precheck.",
        }

    async def check(self, payload):
        self.checked = payload
        return {
            "decision": {"action": self.release_action},
            "final_response": payload["response"] if self.release_action == "allow" else "Held for review.",
        }


def settings(*, fail_closed=True):
    return SimpleNamespace(
        integration_timeout_seconds=1.0,
        graphrag_url="http://127.0.0.1:3100",
        diagnostics_url="http://127.0.0.1:8200",
        controlplane_url="http://127.0.0.1:8100",
        integration_fail_closed=fail_closed,
    )


def test_integrated_analysis_releases_only_after_controlplane_check():
    graph = FakeGraph()
    diagnostics = FakeDiagnostics()
    controlplane = FakeControlPlane()
    orchestrator = IndustrialIntegrationOrchestrator(
        settings=settings(), graph=graph, diagnostics=diagnostics, controlplane=controlplane,
    )
    request = IntegratedAnalysisRequest(
        query="Assess Pump-102",
        diagnostic={"domain": "bearing", "inputs": {"signal": [0.1, 0.2]}},
    )

    result = asyncio.run(
        orchestrator.analyze(request, principal_id="engineer-1", organization_id="apel")
    )

    assert result.released is True
    assert result.status == "released"
    assert "bearing wear" in result.final_response
    assert controlplane.checked["context"]
    assert diagnostics.payload["run_context"]["source"] == "sovereign-ai"
    assert graph.verification_mode == "thorough"


def test_precheck_hold_prevents_evidence_calls():
    graph = FakeGraph()
    controlplane = FakeControlPlane(precheck_action="block")
    orchestrator = IndustrialIntegrationOrchestrator(
        settings=settings(), graph=graph, diagnostics=FakeDiagnostics(), controlplane=controlplane,
    )

    result = asyncio.run(
        orchestrator.analyze(
            IntegratedAnalysisRequest(query="Expose credentials", include_graph_evidence=True),
            principal_id="engineer-1",
            organization_id="apel",
        )
    )

    assert result.released is False
    assert result.status == "precheck_block"
    assert graph.called is False


def test_required_evidence_service_failure_closes_workflow():
    orchestrator = IndustrialIntegrationOrchestrator(
        settings=settings(fail_closed=True),
        graph=FakeGraph(fail=True),
        diagnostics=FakeDiagnostics(),
        controlplane=FakeControlPlane(),
    )

    with pytest.raises(IntegrationServiceError, match="graph-rag"):
        asyncio.run(
            orchestrator.analyze(
                IntegratedAnalysisRequest(query="Assess Pump-102"),
                principal_id="engineer-1",
                organization_id="apel",
            )
        )


def test_public_service_urls_are_rejected():
    with pytest.raises(ValueError, match="internal service address"):
        validate_internal_service_url("https://example.com")

    assert validate_internal_service_url("http://graphrag:3100") == "http://graphrag:3100"
