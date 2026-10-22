"""Released specialist results create one governed, cited report in the same task."""
import asyncio
import json

from docx import Document
import pytest
from sqlalchemy import select

from app.api import tasks
from app.agent.state import RunStatus
from app.core.database import ArtifactRecord, ToolReceiptRecord, HumanApprovalRecord
from app.identity.dependencies import development_principal
from app.identity.models import Permission
from app.integrations.models import IntegratedAnalysisResponse
from diagnostic_fixtures import payload as diagnostic_payload
from test_checkpoints import storage


GOAL = "SYNTHETIC TEST: Diagnose Pump-102 process pressure using document evidence, then generate a maintenance report."
ANSWER = "Pump-102 pressure was 5.1 bar [log-1]. Detected process deviation; root-cause candidate='pressure' [diag-1]."


def setup_case(root, monkeypatch, *, released=True):
    diagnostic = diagnostic_payload("process")
    diagnostic["inputs"]["channel_units"] = ["normalized"] * len(diagnostic["inputs"]["channel_names"])
    diagnostic["run_context"].setdefault("metadata", {}).update(asset_id="Pump-102", provenance="synthetic")
    (root / "readings.json").write_text(json.dumps(diagnostic), encoding="utf8")
    calls = []
    class Services:
        def __init__(self, **kwargs): pass
        async def analyze(self, payload, **kwargs):
            calls.append(payload)
            return IntegratedAnalysisResponse(run_id="specialist-run", released=released,
                status="released" if released else "held_human_review", final_response=ANSWER,
                precheck={"decision": {"action": "allow"}},
                controlplane={"decision": {"action": "allow" if released else "human_review"}},
                graph_evidence={"status": "grounded", "chunks": [
                    {"id": "sop-1", "content": "Isolation requires two-person authorization.",
                     "metadata": {"source_document_id": "SOP-017-R4", "document_status": "current", "document_type": "maintenance_sop", "revision": "4"}},
                    {"id": "old-sop", "content": "OBSOLETE LIMIT MUST NOT APPEAR",
                     "metadata": {"document_status": "superseded", "document_type": "maintenance_sop"}}]},
                diagnostic={"domain": "process", "decision": "diagnose", "document_linkage": "linked",
                    "evidence": [{"evidence_id": "diag-1", "statement": "Detected process deviation; root-cause candidate='pressure'."}]},
                service_status={"graph-rag": "available", "diagnostics": "available", "controlplane": "available", "model": "available"})
    monkeypatch.setattr(tasks, "IndustrialIntegrationOrchestrator", Services)
    return tasks.CreateTaskRequest(request=GOAL, attachments=["readings.json"]), calls


def test_released_industrial_task_creates_report_with_receipt_once(storage, monkeypatch):
    factory, root = storage
    payload, calls = setup_case(root, monkeypatch)
    with factory() as db:
        state = asyncio.run(tasks._execute_task(payload, db, development_principal()))
        assert state.status == RunStatus.completed, state
        assert state.lifecycle.execution_complete and len(calls) == 1
        assert len(state.artifacts) == len(state.tool_records) == 1
        record = db.get(ArtifactRecord, state.artifacts[0]["id"])
        assert record.run_id == state.id and record.owner_id == state.principal_id
        receipt = db.get(ToolReceiptRecord, state.tool_records[0]["operation_id"])
        assert receipt and receipt.run_id == state.id
        paragraphs = "\n".join(p.text for p in Document(root / "artifacts" / record.path).paragraphs)
        assert ANSWER in paragraphs and "SYNTHETIC TEST DATA" in paragraphs
        assert "two-person authorization" in paragraphs and "[sop-1]" in paragraphs
        assert "Normalized diagnostic channels cannot be compared numerically" in paragraphs
        assert "OBSOLETE LIMIT MUST NOT APPEAR" not in paragraphs
        assert state.sources[0]["file"] == "SOP-017-R4"
        assert state.runtime_metrics["timing"]["llm_call_count"] == 0
        # Re-entering finalization formats neither the report nor the assessment again.
        again = asyncio.run(tasks._finalize_task(state, payload, db, development_principal()))
        assert again.status == RunStatus.completed and again.final_response == ANSWER
        assert len(list(db.scalars(select(ArtifactRecord)))) == 1
        assert len(list(db.scalars(select(ToolReceiptRecord)))) == 1


def test_human_review_cannot_export_unreleased_assessment(storage, monkeypatch):
    factory, root = storage
    payload, _ = setup_case(root, monkeypatch, released=False)
    with factory() as db:
        state = asyncio.run(tasks._execute_task(payload, db, development_principal()))
        assert state.status == RunStatus.waiting_for_approval
        assert not state.artifacts and not state.tool_records
        assert not list(db.scalars(select(ArtifactRecord)))


@pytest.mark.parametrize("permission", [Permission.artifact_create, Permission.tool_execute])
def test_report_respects_current_permissions(storage, monkeypatch, permission):
    factory, root = storage
    payload, _ = setup_case(root, monkeypatch)
    principal = development_principal()
    principal.permissions.remove(permission)
    with factory() as db:
        state = asyncio.run(tasks._execute_task(payload, db, principal))
        assert state.status == RunStatus.failed and not state.artifacts
        assert not list(db.scalars(select(ToolReceiptRecord)))


def test_report_obeys_action_approval_policy(storage, monkeypatch):
    from app.governance.action_guard import ActionGuard
    from app.integrations import maintenance_report
    factory, root = storage
    payload, _ = setup_case(root, monkeypatch)
    guard = ActionGuard(tasks.get_settings().tools_config)
    guard.tools["generate_docx"]["risk"] = "HIGH"
    monkeypatch.setattr(maintenance_report, "ActionGuard", lambda *args: guard)
    with factory() as db:
        state = asyncio.run(tasks._execute_task(payload, db, development_principal()))
        assert state.status == RunStatus.waiting_for_approval and not state.artifacts
        approval = db.get(HumanApprovalRecord, state.tool_records[0]["approval_id"])
        assert approval.status == "pending" and approval.run_id == state.id
        assert not list(db.scalars(select(ArtifactRecord)))


def test_requested_sop_comparison_cannot_complete_without_current_sop(storage, monkeypatch):
    factory, root = storage
    payload, _ = setup_case(root, monkeypatch)
    payload.request += " Compare with the current SOP."
    # The adapter must require a procedure, not just any retrieved document.
    from app.integrations import maintenance_report
    real_create = maintenance_report.create_maintenance_report
    async def missing_sop(state, *args):
        state.evidence_records = []
        await real_create(state, *args)
    monkeypatch.setattr(maintenance_report, "create_maintenance_report", missing_sop)
    with factory() as db:
        state = asyncio.run(tasks._execute_task(payload, db, development_principal()))
        assert state.status == RunStatus.failed and not state.artifacts
        assert "current SOP was not retrieved" in state.warnings[-1]


def test_mismatched_diagnostic_asset_cannot_create_report(storage, monkeypatch):
    factory, root = storage
    payload, _ = setup_case(root, monkeypatch)
    data = json.loads((root / "readings.json").read_text())
    data["run_context"]["metadata"]["asset_id"] = "Pump-999"
    (root / "readings.json").write_text(json.dumps(data))
    with factory() as db:
        state = asyncio.run(tasks._execute_task(payload, db, development_principal()))
        assert state.status == RunStatus.failed and not state.artifacts
        assert "asset identity" in state.warnings[-1]
