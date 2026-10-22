"""Safe continuation preserves results and never retries an uncertain action."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api import tasks, governance
from app.agent.state import RunStatus
from app.agent.checkpoints import CheckpointWriter, checkpoint_context, recover_expired_tasks
from app.agent.resume import prepare_resume
from app.core.database import AgentRunRecord, ArtifactRecord, HumanApprovalRecord, TaskExecutionRecord, ToolReceiptRecord
from app.identity.dependencies import development_principal
from app.identity.models import Permission
from app.llm.base import ModelGenerationCancelled
from app.main import app
from app.tools.registry import create_agent_registry, create_numeric_registry
from test_checkpoints import storage, saved, initial, events
from test_structured_plan import Script, PLAN, GOAL


def interrupted(factory, monkeypatch, *, request=GOAL, plan=PLAN, scope="NUMERIC", stop="tool_completed"):
    script = Script(plan)
    monkeypatch.setattr(tasks, "configured_local_provider", lambda *a, **k: SimpleNamespace(model="test", provider=script))
    async def run():
        cancel = asyncio.Event()
        async def emit(kind, data):
            if kind == stop:
                cancel.set()
        with factory() as db:
            with pytest.raises(ModelGenerationCancelled):
                await tasks._execute_task(tasks.CreateTaskRequest(request=request, tool_scope=scope), db,
                    development_principal(), emit, cancel)
            return saved(db, db.scalar(select(AgentRunRecord.id))).id
    return asyncio.run(run()), script


def reconcile(factory, run_id, monkeypatch, *, principal=None):
    with factory() as db:
        state = saved(db, run_id)
        execution = db.get(TaskExecutionRecord, run_id)
        tools = create_numeric_registry() if json.loads(execution.payload_json)["tool_scope"] == "NUMERIC" else create_agent_registry(tasks.get_settings(), db, principal or development_principal(), tasks._task_scope(db.get(AgentRunRecord, run_id)))
        return prepare_resume(db, tasks.get_settings(), state, execution, principal or development_principal(), state.checkpoint.revision, tools)


def test_resume_plan_only_executes_remaining_steps_with_same_id(storage, monkeypatch):
    factory, _ = storage
    run_id, script = interrupted(factory, monkeypatch)
    with TestClient(app) as client:
        before = client.get(f"/api/tasks/{run_id}").json()
        response = client.post(f"/api/tasks/{run_id}/resume", json={"checkpoint_revision": before["checkpoint"]["revision"]})
        assert response.status_code == 202, response.text
        stream = events(client, run_id)
        result = stream[-1]["payload"]["result"]
        assert result["id"] == run_id and result["status"] == "completed", result
        assert len(result["tool_records"]) == 3
        assert [item["payload"]["tool"] for item in stream if item["type"] == "tool_started"] == ["calculate", "convert_units"]
        assert result["checkpoint"]["data"]["resume_attempts"] == 1
        assert result["runtime_metrics"]["active_seconds"] >= before["runtime_metrics"]["active_seconds"]
        assert client.post(f"/api/tasks/{run_id}/resume", json={"checkpoint_revision": before["checkpoint"]["revision"]}).status_code == 409
    assert len(script.prompts) == 1


def test_resume_rejects_current_permission_loss_before_execution(storage, monkeypatch):
    factory, _ = storage
    run_id, _ = interrupted(factory, monkeypatch)
    principal = development_principal()
    principal.permissions.remove(Permission.tool_execute)
    with pytest.raises(HTTPException) as error:
        reconcile(factory, run_id, monkeypatch, principal=principal)
    assert error.value.status_code == 403
    with factory() as db:
        assert len(saved(db, run_id).tool_records) == 1
        assert db.get(TaskExecutionRecord, run_id).phase == "finished"


def test_stale_revision_and_unknown_outcome_are_not_replayed(storage, monkeypatch):
    factory, _ = storage
    run_id, _ = interrupted(factory, monkeypatch)
    with factory() as db:
        state = saved(db, run_id)
        with pytest.raises(HTTPException) as error:
            prepare_resume(db, tasks.get_settings(), state, db.get(TaskExecutionRecord, run_id), development_principal(), state.checkpoint.revision - 1, create_numeric_registry())
        assert error.value.detail["code"] == "RESUME_STALE_REVISION"
        # A started tool without either a result or receipt cannot be repeated.
        record = state.tool_records[-1]
        db.delete(db.get(ToolReceiptRecord, record["operation_id"]))
        record.update(success=None, execution_status="unknown")
        record.pop("output")
        writer = CheckpointWriter(db, state, development_principal())
        writer.claim_resume(state.checkpoint.revision)
        writer.save(state, "unknown_test", final=True)
    with pytest.raises(HTTPException) as error:
        reconcile(factory, run_id, monkeypatch)
    assert error.value.detail["code"] == "RESUME_ACTION_OUTCOME_UNKNOWN"


def test_lease_claim_is_atomic_and_stale_worker_is_fenced(storage):
    factory, _ = storage
    with factory() as db:
        state = initial(db)
        writer = CheckpointWriter(db, state, development_principal())
        writer.claim()
        writer.stop(RunStatus.cancelled, "Paused for test")
        revision = state.checkpoint.revision
        winner = CheckpointWriter(db, state, development_principal())
        winner.claim_resume(revision)
        winner.save(state, "resume_accepted")
    with factory() as db:
        loser = CheckpointWriter(db, state.model_copy(deep=True), development_principal())
        with pytest.raises(RuntimeError):
            loser.claim_resume(revision)
        with pytest.raises(RuntimeError):
            loser.save(state, "stale_worker_write")


def test_changed_read_input_blocks_resume(storage, monkeypatch):
    factory, root = storage
    (root / "input.txt").write_text("Original content")
    plan = {"action": "plan", "reason_summary": "Read then inspect", "steps": [
        {"id": "read", "tool": "read_file", "arguments": {"path": "input.txt"}, "depends_on": []},
        {"id": "metadata", "tool": "get_file_metadata", "arguments": {"path": "input.txt"}, "depends_on": ["read"]}]}
    run_id, _ = interrupted(factory, monkeypatch, request="Use read_file then get_file_metadata for input.txt", plan=plan, scope="ALL")
    (root / "input.txt").write_text("Changed content")
    with pytest.raises(HTTPException) as error:
        reconcile(factory, run_id, monkeypatch)
    assert error.value.detail["code"] == "RESUME_INPUT_CHANGED"


def test_approval_continues_plan_without_repeating_approved_write(storage, monkeypatch):
    factory, root = storage
    settings = tasks.get_settings()
    monkeypatch.setattr(governance, "get_settings", lambda: settings)
    (root / "input.txt").write_text("Copy this exact content")
    plan = {"action": "plan", "reason_summary": "Read, write with approval, inspect", "steps": [
        {"id": "read", "tool": "read_file", "arguments": {"path": "input.txt"}, "depends_on": []},
        {"id": "write", "tool": "write_file", "arguments": {"path": "output.txt", "content": {"$ref": "read.text"}}, "depends_on": ["read"]},
        {"id": "metadata", "tool": "get_file_metadata", "arguments": {"path": "output.txt"}, "depends_on": ["write"]}]}
    script = Script(plan, {"final_response": "The file was copied and its metadata checked.", "reason_summary": "Verified results"})
    monkeypatch.setattr(tasks, "configured_local_provider", lambda *a, **k: SimpleNamespace(model="test", provider=script))
    with TestClient(app) as client:
        started = client.post("/api/tasks/start", json={"request": "Use read_file input.txt, write_file output.txt with the exact contents, then get_file_metadata output.txt"}).json()
        run_id = started["task_id"]
        paused = events(client, run_id)[-1]["payload"]["result"]
        assert paused["status"] == "waiting_for_approval", paused
        approval_id = paused["tool_records"][-1]["approval_id"]
        pending = client.post(f"/api/tasks/{run_id}/resume", json={"checkpoint_revision": paused["checkpoint"]["revision"]})
        assert pending.status_code == 409 and "RESUME_APPROVAL_PENDING" in pending.text
        approved = client.post(f"/api/approvals/{approval_id}", json={"approve": True})
        assert approved.status_code == 200, approved.text
        assert approved.json()["continuation"]["status"] == "accepted", approved.json()
        result = events(client, run_id)[-1]["payload"]["result"]
        assert result["status"] == "completed", result
        assert [item["tool"] for item in result["tool_records"]] == ["read_file", "write_file", "get_file_metadata"]
        assert result["tool_records"][1]["reconciliation"]["type"] == "approval"
        assert (root / "output.txt").read_text() == "Copy this exact content"
        assert client.post(f"/api/approvals/{approval_id}", json={"approve": True}).status_code == 409
        with factory() as db:
            assert len(db.scalars(select(ToolReceiptRecord).where(ToolReceiptRecord.run_id == run_id)).all()) == 3


@pytest.mark.parametrize("case,code", [
    ("disabled_tool", "RESUME_TOOL_DISABLED"),
    ("payload", "RESUME_PAYLOAD_CHANGED"),
    ("time_budget", "RESUME_TIME_BUDGET_EXHAUSTED"),
    ("attempts", "RESUME_ATTEMPT_LIMIT"),
    ("receipt_hash", "RESUME_RECEIPT_INTEGRITY_FAILED"),
    ("action_hash", "RESUME_ACTION_CHANGED"),
])
def test_resume_denials_leave_original_progress_intact(storage, monkeypatch, case, code):
    factory, root = storage
    run_id, _ = interrupted(factory, monkeypatch)
    with factory() as db:
        state = saved(db, run_id)
        if case == "disabled_tool":
            config = root / "tools.yaml"
            config.write_text(tasks.get_settings().tools_config.read_text().replace("summarize_numbers: {risk: LOW, enabled: true}", "summarize_numbers: {risk: LOW, enabled: false}"))
            settings = tasks.get_settings().model_copy(update={"tools_config": config})
            monkeypatch.setattr(tasks, "get_settings", lambda: settings)
        elif case == "payload":
            execution = db.get(TaskExecutionRecord, run_id)
            payload = json.loads(execution.payload_json)
            payload["tool_scope"] = "ALL"
            execution.payload_json = json.dumps(payload)
            db.commit()
        elif case == "receipt_hash":
            receipt = db.get(ToolReceiptRecord, state.tool_records[0]["operation_id"])
            receipt.result_json = '{"success":true,"output":{"mean":999}}'
            db.commit()
        else:
            if case == "time_budget":
                state.runtime_metrics["active_seconds"] = 10000
            elif case == "attempts":
                state.checkpoint.data["resume_attempts"] = 3
            elif case == "action_hash":
                state.tool_records[0]["arguments"] = {"values": [999]}
            writer = CheckpointWriter(db, state, development_principal())
            writer.claim_resume(state.checkpoint.revision)
            writer.save(state, "test_saved_change", final=True)
    with pytest.raises(HTTPException) as error:
        reconcile(factory, run_id, monkeypatch)
    assert error.value.detail["code"] == code
    with factory() as db:
        assert len(saved(db, run_id).tool_records) == 1


@pytest.mark.parametrize("enabled,code", [(False, "RESUME_IDENTITY_INVALID"), (True, "RESUME_TOOL_ACCESS_REVOKED")])
def test_local_identity_is_reloaded_and_disabled_owner_cannot_resume(storage, monkeypatch, enabled, code):
    from app.core.database import UserRecord
    factory, _ = storage
    run_id, _ = interrupted(factory, monkeypatch)
    settings = tasks.get_settings().model_copy(update={"auth_mode": "local"})
    monkeypatch.setattr(tasks, "get_settings", lambda: settings)
    owner = development_principal()
    with factory() as db:
        db.add(UserRecord(id=owner.user_id, email=owner.email, email_normalized=owner.email,
            display_name=owner.display_name, organization_id="*", workspace_ids_json='["*"]',
            department_ids_json='["*"]', roles_json='[]', permissions_json='["task.read","task.create"]',
            clearance="RESTRICTED", password_hash="unused", enabled=enabled))
        db.commit()
    with pytest.raises(HTTPException) as error:
        reconcile(factory, run_id, monkeypatch)
    assert error.value.detail["code"] == code


def test_other_user_cannot_take_ownership_even_with_admin_permissions(storage, monkeypatch):
    factory, _ = storage
    run_id, _ = interrupted(factory, monkeypatch)
    stranger = development_principal().model_copy(update={"user_id": "other-admin"})
    with pytest.raises(HTTPException) as error:
        reconcile(factory, run_id, monkeypatch, principal=stranger)
    assert error.value.detail["code"] == "RESUME_OWNER_REQUIRED"


def test_resume_rechecks_permissions_before_next_tool(storage, monkeypatch):
    factory, _ = storage
    run_id, _ = interrupted(factory, monkeypatch)
    from app.agent import resume
    original = resume.revalidate_principal
    checks = []
    def revoked(db, settings, principal, state):
        checks.append(state.checkpoint.boundary)
        # The request is accepted, but permission is lost before its first tool.
        if len(checks) >= 4:
            principal.permissions = [item for item in principal.permissions if item != Permission.tool_execute]
        return original(db, settings, principal, state)
    monkeypatch.setattr(resume, "revalidate_principal", revoked)
    with TestClient(app) as client:
        state = client.get(f"/api/tasks/{run_id}").json()
        response = client.post(f"/api/tasks/{run_id}/resume", json={"checkpoint_revision": state["checkpoint"]["revision"]})
        assert response.status_code == 202
        result = events(client, run_id)[-1]["payload"]["result"]
        assert result["status"] == "failed"
        assert len(result["tool_records"]) == 1
    assert len(checks) >= 4



def test_adaptive_resume_keeps_observations_and_decision_budget(storage, monkeypatch):
    factory, _ = storage
    script = Script({"action": "adaptive", "reason_summary": "Decide after results.", "steps": []},
        {"action": "tool", "tool_name": "calculate", "arguments": {"expression": "2+3"}, "reason_summary": "Calculate"},
        {"action": "tool", "tool_name": "convert_units", "arguments": {"value": 5, "from_unit": "kPa", "to_unit": "bar"}, "reason_summary": "Convert actual output"},
        {"action": "complete", "final_response": "5 kPa equals 0.05 bar.", "reason_summary": "Verified outputs"})
    monkeypatch.setattr(tasks, "configured_local_provider", lambda *a, **k: SimpleNamespace(model="test", provider=script))
    async def pause():
        cancel = asyncio.Event()
        async def emit(kind, data):
            if kind == "tool_completed":
                cancel.set()
        with factory() as db:
            with pytest.raises(ModelGenerationCancelled):
                await tasks._execute_task(tasks.CreateTaskRequest(request="Use calculate then convert_units", tool_scope="NUMERIC"), db, development_principal(), emit, cancel)
            return db.scalar(select(AgentRunRecord.id))
    run_id = asyncio.run(pause())
    with TestClient(app) as client:
        state = client.get(f"/api/tasks/{run_id}").json()
        assert state["runtime_metrics"]["decisions_used"] == 1
        response = client.post(f"/api/tasks/{run_id}/resume", json={"checkpoint_revision": state["checkpoint"]["revision"]})
        assert response.status_code == 202, response.text
        result = events(client, run_id)[-1]["payload"]["result"]
        assert result["status"] == "completed", result
        assert [item["tool"] for item in result["tool_records"]] == ["calculate", "convert_units"]
        assert result["runtime_metrics"]["decisions_used"] == 3
        assert '"expression":"2+3"' in script.prompts[2].replace(" ", "") and '"result":5.0' in script.prompts[2].replace(" ", "")


def test_direct_resume_preserves_generated_answer_without_new_model_call(storage, monkeypatch):
    from app.agent import orchestrator
    from app.llm.base import GenerationChunk
    factory, _ = storage
    calls = []
    class Stream:
        async def stream(self, *args, **kwargs):
            calls.append(args)
            yield GenerationChunk(text="Cavitation forms vapor bubbles in a low-pressure liquid.", model="test", provider="test", done=True)
    monkeypatch.setattr(orchestrator, "configured_local_provider", lambda *a, **k: SimpleNamespace(model="test", provider=Stream()))
    async def pause():
        cancel = asyncio.Event()
        async def emit(kind, data):
            if kind == "generation_completed":
                cancel.set()
        with factory() as db:
            with pytest.raises(ModelGenerationCancelled):
                await tasks._execute_task(tasks.CreateTaskRequest(request="Explain pump cavitation", chat_mode="GENERAL"), db, development_principal(), emit, cancel)
            return db.scalar(select(AgentRunRecord.id))
    run_id = asyncio.run(pause())
    with TestClient(app) as client:
        state = client.get(f"/api/tasks/{run_id}").json()
        assert state["checkpoint"]["data"]["draft_response"].startswith("Cavitation")
        response = client.post(f"/api/tasks/{run_id}/resume", json={"checkpoint_revision": state["checkpoint"]["revision"]})
        assert response.status_code == 202, response.text
        result = events(client, run_id)[-1]["payload"]["result"]
        assert result["status"] == "completed" and result["final_response"].startswith("Cavitation"), result
    assert len(calls) == 1


@pytest.mark.parametrize("outcome", ["confirmed", "changed_output", "missing_receipt"])
def test_crashed_approval_uses_committed_receipt_and_repairs_decision(storage, monkeypatch, outcome):
    from app.tools.file_tools import WriteFileTool, SafeWorkspace
    from app.tools.registry import ToolRegistry
    factory, root = storage
    settings = tasks.get_settings()
    monkeypatch.setattr(governance, "get_settings", lambda: settings)
    plan = {"action": "plan", "reason_summary": "Write and inspect an approved file", "steps": [
        {"id": "write", "tool": "write_file", "arguments": {"path": "approved.txt", "content": "written once"}, "depends_on": []},
        {"id": "metadata", "tool": "get_file_metadata", "arguments": {"path": "approved.txt"}, "depends_on": ["write"]}]}
    script = Script(plan, {"final_response": "The approved file was written.", "reason_summary": "Confirmed receipt"})
    monkeypatch.setattr(tasks, "configured_local_provider", lambda *a, **k: SimpleNamespace(model="test", provider=script))
    with factory() as db:
        paused = asyncio.run(tasks._execute_task(tasks.CreateTaskRequest(request="Use write_file to create approved.txt containing written once then get_file_metadata", chat_mode="CONTROLLED"), db, development_principal()))
        approval_id = paused.tool_records[0]["approval_id"]
    writes = []
    class CrashWrite(WriteFileTool):
        async def execute(self, args):
            writes.append(args)
            result = await super().execute(args)
            assert result.success
            raise asyncio.CancelledError("Simulate crash after durable receipt")
    registry = ToolRegistry()
    registry.register(CrashWrite(SafeWorkspace(root)))
    monkeypatch.setattr(governance, "create_agent_registry", lambda *a, **k: registry)
    with factory() as db:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(governance.decide_approval(approval_id, governance.ApprovalDecision(approve=True), development_principal(), db))
    if outcome == "changed_output":
        (root / "approved.txt").write_text("Changed externally")
    elif outcome == "missing_receipt":
        with factory() as db:
            db.delete(db.get(ToolReceiptRecord, paused.tool_records[0]["operation_id"]))
            db.commit()
    with TestClient(app) as client:
        state = client.get(f"/api/tasks/{paused.id}").json()
        response = client.post(f"/api/tasks/{paused.id}/resume", json={"checkpoint_revision": state["checkpoint"]["revision"]})
        if outcome != "confirmed":
            assert response.status_code == 409, response.text
            assert ("RESUME_OUTPUT_CHANGED" if outcome == "changed_output" else "RESUME_APPROVAL_OUTCOME_UNKNOWN") in response.text
            assert len(writes) == 1
            return
        assert response.status_code == 202, response.text
        result = events(client, paused.id)[-1]["payload"]["result"]
        assert result["status"] == "completed", result
        assert result["tool_records"][0]["reconciliation"]["approval_id"] == approval_id
        assert client.get(f"/api/approvals/{approval_id}").json()["status"] == "executed"
    assert len(writes) == 1 and (root / "approved.txt").read_text() == "written once"



def test_unstarted_task_can_resume_without_restarting_completed_work(storage, monkeypatch):
    factory, _ = storage
    script = Script(PLAN)
    monkeypatch.setattr(tasks, "configured_local_provider", lambda *a, **k: SimpleNamespace(model="test", provider=script))
    with factory() as db:
        state = initial(db, request=GOAL)
        lease = db.get(TaskExecutionRecord, state.id)
        lease.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()
        assert recover_expired_tasks(db, state.id) == 1
    with TestClient(app) as client:
        state = client.get(f"/api/tasks/{state.id}").json()
        response = client.post(f"/api/tasks/{state['id']}/resume", json={"checkpoint_revision": state["checkpoint"]["revision"]})
        assert response.status_code == 202, response.text
        result = events(client, state["id"])[-1]["payload"]["result"]
        assert result["status"] == "completed" and len(result["tool_records"]) == 3, result
    assert len(script.prompts) == 1


def test_concurrent_approvals_execute_exactly_one_write(storage, monkeypatch):
    from app.tools.file_tools import WriteFileTool, SafeWorkspace
    from app.tools.registry import ToolRegistry
    factory, root = storage
    settings = tasks.get_settings()
    monkeypatch.setattr(governance, "get_settings", lambda: settings)
    script = Script({"action": "tool", "tool_name": "write_file", "arguments": {"path": "single.txt", "content": "one write"}, "reason_summary": "Write"})
    monkeypatch.setattr(tasks, "configured_local_provider", lambda *a, **k: SimpleNamespace(model="test", provider=script))
    with factory() as db:
        state = asyncio.run(tasks._execute_task(tasks.CreateTaskRequest(request="Use write_file single.txt", chat_mode="CONTROLLED"), db, development_principal()))
        approval_id = state.tool_records[0]["approval_id"]
    async def accepted(*args):
        return tasks.StartTaskResponse(task_id=state.id)
    monkeypatch.setattr(tasks, "schedule_resume", accepted)
    writes = []
    async def race():
        entered, release = asyncio.Event(), asyncio.Event()
        class YieldWrite(WriteFileTool):
            async def execute(self, args):
                writes.append(args)
                result = await super().execute(args)
                entered.set()
                await release.wait()
                return result
        with factory() as first, factory() as second:
            registry = ToolRegistry()
            registry.register(YieldWrite(SafeWorkspace(root)))
            monkeypatch.setattr(governance, "create_agent_registry", lambda *a, **k: registry)
            pending = asyncio.create_task(governance.decide_approval(approval_id, governance.ApprovalDecision(approve=True), development_principal(), first))
            await asyncio.wait_for(entered.wait(), 2)
            with pytest.raises(HTTPException) as duplicate:
                await governance.decide_approval(approval_id, governance.ApprovalDecision(approve=True), development_principal(), second)
            assert duplicate.value.status_code == 409
            release.set()
            assert (await pending)["status"] == "executed"
    asyncio.run(race())
    assert len(writes) == 1 and (root / "single.txt").read_text() == "one write"



def test_resume_preserves_original_workspace_when_membership_order_changes(storage, monkeypatch):
    from app.agent.checkpoints import accept_task
    from app.identity.dependencies import get_current_principal
    factory, _ = storage
    owner = development_principal().model_copy(update={"user_id": "owner", "organization_id": "org-a", "workspace_ids": ["original", "current"]})
    current = owner.model_copy(update={"workspace_ids": ["current", "original"]})
    app.dependency_overrides[get_current_principal] = lambda: current
    plan = {"action": "plan", "reason_summary": "Compute and report", "steps": [
        {"id": "calc", "tool": "calculate", "arguments": {"expression": "2+3"}, "depends_on": []},
        {"id": "report", "tool": "generate_docx", "arguments": {"filename": "scope.docx", "title": "Result", "sections": [{"heading": "Value", "content": "{{calc.result}}"}]}, "depends_on": ["calc"]}]}
    script = Script(plan, {"final_response": "Calculated 5 and created the report.", "reason_summary": "Actual outputs"})
    monkeypatch.setattr(tasks, "configured_local_provider", lambda *a, **k: SimpleNamespace(model="test", provider=script))
    with factory() as db:
        payload = tasks.CreateTaskRequest(request="Use calculate then generate_docx for a report", chat_mode="CONTROLLED")
        state = tasks._accepted_state(payload, "workspace-task")
        accept_task(db, state, payload.model_dump(mode="json"), owner)
        lease = db.get(TaskExecutionRecord, state.id)
        lease.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()
        recover_expired_tasks(db, state.id)
    with TestClient(app) as client:
        state = client.get("/api/tasks/workspace-task").json()
        response = client.post("/api/tasks/workspace-task/resume", json={"checkpoint_revision": state["checkpoint"]["revision"]})
        assert response.status_code == 202, response.text
        result = events(client, "workspace-task")[-1]["payload"]["result"]
        assert result["status"] == "completed" and result["workspace_id"] == "original", result
    with factory() as db:
        assert db.scalar(select(ArtifactRecord)).workspace_id == "original"



def test_executed_approval_cannot_hide_missing_side_effect_receipt(storage, monkeypatch):
    factory, _ = storage
    settings = tasks.get_settings()
    monkeypatch.setattr(governance, "get_settings", lambda: settings)
    script = Script({"action": "tool", "tool_name": "write_file", "arguments": {"path": "missing.txt", "content": "confirmed by approval only"}, "reason_summary": "Write"})
    monkeypatch.setattr(tasks, "configured_local_provider", lambda *a, **k: SimpleNamespace(model="test", provider=script))
    with factory() as db:
        paused = asyncio.run(tasks._execute_task(tasks.CreateTaskRequest(request="Use write_file missing.txt", chat_mode="CONTROLLED"), db, development_principal()))
        record = paused.tool_records[0]
        approval = db.get(HumanApprovalRecord, record["approval_id"])
        approval.status = approval.execution_status = "executed"
        approval.result_json = json.dumps({"success": True, "output": {"bytes": 26}, "error": None, "generated_files": ["missing.txt"]})
        db.commit()
    with pytest.raises(HTTPException) as error:
        reconcile(factory, paused.id, monkeypatch)
    assert error.value.detail["code"] == "RESUME_SIDE_EFFECT_RECEIPT_REQUIRED"



@pytest.mark.parametrize("missing", ["execution", "invalid_payload"])
def test_legacy_or_invalid_saved_request_is_rejected_without_starting_worker(storage, missing):
    factory, _ = storage
    with factory() as db:
        state = initial(db)
        execution = db.get(TaskExecutionRecord, state.id)
        if missing == "execution":
            db.delete(execution)
        else:
            execution.payload_json = '{"invalid":true}'
        db.commit()
        with pytest.raises(HTTPException) as error:
            asyncio.run(tasks.schedule_resume(db, development_principal(), state, state.checkpoint.revision))
        assert error.value.status_code == 409
        assert error.value.detail["code"] == ("RESUME_CHECKPOINT_REQUIRED" if missing == "execution" else "RESUME_PAYLOAD_INVALID")
