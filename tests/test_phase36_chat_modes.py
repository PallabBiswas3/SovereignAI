from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.state import AgentPlan, AgentRunState, RunStatus
from app.api.tasks import (
    CreateTaskRequest, _authorized_evidence_limit, _authorized_generation_prompt,
    _run_authorized_task,
)
from app.core.config import get_settings
from app.core.database import Base
from app.demo.apel import ApelDemoService
from app.identity.provider import LocalIdentityProvider
from app.orchestration.chat_mode import (
    ChatMode, ChatModeSelector, extract_asset_references,
    requires_structured_asset_assessment, system_prompt_for_mode,
)
from app.router.model_registry import ModelRegistry
from app.router.model_router import ModelRouter
from app.evidence.models import EvidenceSource
from app.rag.retrieval import RetrievedChunk
from app.workflows.inspection import InspectionReading, InspectionWorkflow


ROOT = Path(__file__).resolve().parents[1]


def test_chat_mode_selector_separates_general_authorized_and_controlled_work() -> None:
    selector = ChatModeSelector()
    assert selector.select(ChatMode.automatic, "What causes pump cavitation?").selected == ChatMode.general
    assert selector.select(ChatMode.automatic, "Assess Pump-102 using current telemetry").selected == ChatMode.authorized
    assert selector.select(ChatMode.automatic, "Prepare a maintenance report").selected == ChatMode.controlled
    assert selector.select(
        ChatMode.automatic, "Summarize this", attachment_count=1,
    ).selected == ChatMode.controlled
    assert selector.select(
        ChatMode.general, "Assess Pump-102 using current telemetry",
    ).selected == ChatMode.general


def test_task_contract_accepts_explicit_chat_mode_and_extracts_exact_asset_tags() -> None:
    request = CreateTaskRequest.model_validate({
        "request": "Compare Pump-102 with Compressor-201", "chat_mode": "AUTHORIZED",
    })
    assert request.chat_mode == ChatMode.authorized
    assert extract_asset_references(request.request) == ["Pump-102", "Compressor-201"]
    assert "general guidance" in system_prompt_for_mode(ChatMode.general)
    assert "AUTHORIZED_CONTEXT" in system_prompt_for_mode(ChatMode.authorized)


def test_asset_condition_intent_does_not_hijack_document_analysis() -> None:
    assert requires_structured_asset_assessment(
        "Assess Pump-102 using current telemetry",
    ) is True
    assert requires_structured_asset_assessment(
        "Compare all authorized vendor proposals against the technical requirements for Compressor-201",
    ) is False


def test_cross_department_briefing_gets_bounded_expanded_context_and_grounding_rules() -> None:
    request = (
        "Prepare a management briefing covering operations, maintenance, incident safety, and quality"
    )
    assert _authorized_evidence_limit(request) == 10
    assert _authorized_evidence_limit("Compare vendor proposals for Compressor-201") == 6
    prompt = _authorized_generation_prompt(request, None, [{
        "chunk_id": "C1",
        "document_id": "D1",
        "text": "Incident contained; permit pending.",
        "source": {"file": "Briefing.md", "department": "management"},
        "scores": {"dense": 0.9},
        "telemetry": {"large_internal_diagnostic": "must-not-reach-model"},
        "cache_hit": True,
    }])
    assert "is not evidence that the required action" in prompt
    assert "under 400 words" in prompt
    assert "Incident contained; permit pending." in prompt
    assert "Briefing.md" in prompt
    assert "large_internal_diagnostic" not in prompt
    assert '"scores"' not in prompt
    assert requires_structured_asset_assessment(
        "Analyze the safety permit for Incident-2026-014 near Pump-102",
    ) is False


def test_apel_seed_preserves_revision_and_inspection_rules_do_not_cross_metrics(tmp_path: Path) -> None:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        service = ApelDemoService(session, ROOT / "demo/apel", tmp_path / "apel")
        manifest = service.generate()
    current_sop = next(
        item for item in manifest
        if item["path"].endswith("SOP-MNT-017_Pump_Condition_Monitoring_Rev4.md")
    )
    assert current_sop["revision"] == "Rev 4"
    compressor_documents = [
        item for item in manifest
        if item["path"].endswith("Compressor-201_Technical_Requirements.md")
        or "/Vendor_" in item["path"]
    ]
    assert len(compressor_documents) == 4
    assert all(item["asset_id"] == "Compressor-201" for item in compressor_documents)

    text = (
        "SOP-MNT-017 Rev 4. Investigate vibration above 6.0 mm/s RMS and "
        "initiate controlled shutdown at 9.0 mm/s RMS."
    )
    source = EvidenceSource(id="E1", file="SOP-MNT-017_Rev4.md", text=text, revision="Rev 4")
    vibration = InspectionReading(parameter="Vibration", metric="vibration", value=8.2, unit="mm/s")
    rules = InspectionWorkflow._extract_rules(vibration, text, source)
    assert {(item.rule_type, item.threshold) for item in rules} == {
        ("normal_limit", 6.0), ("shutdown_limit", 9.0),
    }

    temperature = InspectionReading(
        parameter="Bearing temperature", metric="bearing_temperature", value=86.0, unit="°C",
    )
    candidate = RetrievedChunk(chunk_id="C1", text=text, score=1.0, source={})
    assert InspectionWorkflow._candidate_applies(temperature, candidate) is False
    other_asset = RetrievedChunk(
        chunk_id="C2", text="Maximum normal vibration: 4.5 mm/s RMS.", score=1.0,
        source={"asset_id": "Motor-202"},
    )
    assert InspectionWorkflow._candidate_applies(vibration, other_asset, "Pump-102") is False


def test_authorized_mode_injects_only_principal_scoped_asset_context(monkeypatch, tmp_path: Path) -> None:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    class EmptyRetriever:
        def search(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr("app.api.tasks.configured_hybrid_retriever", lambda *_args, **_kwargs: EmptyRetriever())

    with Session(engine) as session:
        ApelDemoService(session, ROOT / "demo/apel", tmp_path / "apel").seed()
        principal = LocalIdentityProvider(session, ROOT / "config/access.yaml").principal_for_user("apel-maint-001")
        assert principal
        payload = CreateTaskRequest(
            request="Assess Pump-102 using current telemetry", chat_mode=ChatMode.authorized,
        )
        selection = ChatModeSelector().select(payload.chat_mode, payload.request)
        state = asyncio.run(_run_authorized_task(
            payload, get_settings(), session, principal, selection,
        ))

    assert state.asset_context and state.asset_context["asset"]["asset_id"] == "Pump-102"
    assert state.runtime_metrics["provider"] == "deterministic-evidence-engine"
    assert state.runtime_metrics["model_invoked"] is False
    assert "TEL-P102-VIB-06" in str(state.final_response)
    assert "RULE-P102-VIBRATION-NORMAL" in str(state.final_response)
    assert "No plant command was issued" in str(state.final_response)
    assert "FIN-XC-926" not in str(state.final_response)
    assert state.context_metrics["authorization_scope"] == "authenticated-principal"


def test_authorized_mode_does_not_inject_asset_values_for_unauthorized_user(monkeypatch, tmp_path: Path) -> None:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    captured: dict[str, object] = {}

    class EmptyRetriever:
        def search(self, *_args, **_kwargs):
            return []

    async def fake_run(self, request, *_args, **kwargs):
        captured["prompt"] = kwargs.get("generation_prompt")
        routing = ModelRouter(ModelRegistry(get_settings().models_config)).route(request)
        return AgentRunState(
            id=str(uuid4()), request=request, status=RunStatus.completed,
            plan=AgentPlan(goal=request, steps=[]), routing=routing,
            final_response="Authorized evidence is insufficient.",
        )

    monkeypatch.setattr("app.api.tasks.configured_hybrid_retriever", lambda *_args, **_kwargs: EmptyRetriever())
    monkeypatch.setattr("app.api.tasks.AgentOrchestrator.run", fake_run)

    with Session(engine) as session:
        ApelDemoService(session, ROOT / "demo/apel", tmp_path / "apel").seed()
        principal = LocalIdentityProvider(session, ROOT / "config/access.yaml").principal_for_user("apel-auditor-001")
        assert principal
        payload = CreateTaskRequest(
            request="Assess Pump-102 using current telemetry", chat_mode=ChatMode.authorized,
        )
        selection = ChatModeSelector().select(payload.chat_mode, payload.request)
        state = asyncio.run(_run_authorized_task(
            payload, get_settings(), session, principal, selection,
        ))

    prompt = str(captured["prompt"])
    assert '"asset_context": null' in prompt
    assert "TEL-P102" not in prompt and "8.2" not in prompt
    assert state.asset_context is None
    assert any("access scope" in warning for warning in state.warnings)


def test_procurement_comparison_uses_asset_scoped_rag_not_telemetry(monkeypatch, tmp_path: Path) -> None:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    captured: dict[str, object] = {}

    class ProcurementRetriever:
        def search(self, request, _limit, *, asset_id=None):
            captured["asset_id"] = asset_id
            return [RetrievedChunk(
                chunk_id="PROC-ATLAS-001",
                text="Atlas proposal: capacity 4900 Nm3/h; discharge pressure 7.5 bar.",
                score=1.0,
                source={"file": "Atlas_Compressor_Proposal.md", "asset_id": "Compressor-201"},
            )]

    async def fake_run(self, request, *_args, **kwargs):
        captured["prompt"] = kwargs.get("generation_prompt")
        routing = ModelRouter(ModelRegistry(get_settings().models_config)).route(request)
        return AgentRunState(
            id=str(uuid4()), request=request, status=RunStatus.completed,
            plan=AgentPlan(goal=request, steps=[]), routing=routing,
            final_response="Compared the authorized proposals.",
        )

    def fail_if_compiled(*_args, **_kwargs):
        raise AssertionError("Procurement analysis must not compile telemetry context")

    monkeypatch.setattr(
        "app.api.tasks.configured_hybrid_retriever",
        lambda *_args, **_kwargs: ProcurementRetriever(),
    )
    monkeypatch.setattr("app.api.tasks.AgentOrchestrator.run", fake_run)
    monkeypatch.setattr("app.api.tasks.AssetContextService.compile", fail_if_compiled)

    request = (
        "Compare all authorized vendor proposals against the technical requirements "
        "for Compressor-201"
    )
    with Session(engine) as session:
        ApelDemoService(session, ROOT / "demo/apel", tmp_path / "apel").seed()
        principal = LocalIdentityProvider(session, ROOT / "config/access.yaml").principal_for_user(
            "apel-proc-001",
        )
        assert principal
        payload = CreateTaskRequest(request=request, chat_mode=ChatMode.authorized)
        selection = ChatModeSelector().select(payload.chat_mode, payload.request)
        state = asyncio.run(_run_authorized_task(
            payload, get_settings(), session, principal, selection,
        ))

    assert captured["asset_id"] == "Compressor-201"
    assert "Atlas proposal" in str(captured["prompt"])
    assert state.asset_context is None
    assert state.context_metrics["resolved_asset_id"] == "Compressor-201"
    assert state.context_metrics["structured_asset_assessment"] is False
