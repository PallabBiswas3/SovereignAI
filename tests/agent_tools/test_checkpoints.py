"""Durable progress survives cancellation, worker loss, and loss of SSE memory."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.agent import checkpoints
from app.agent.checkpoints import CheckpointWriter, accept_task, checkpoint_context, recover_expired_tasks
from app.agent.state import AgentRunState, RunStatus
from app.api import tasks
from app.core.config import get_settings
from app.core.database import AgentRunRecord, ArtifactRecord, Base, TaskCheckpointRecord, TaskExecutionRecord, get_db
from app.core.events import TaskEventBroker
from app.identity.dependencies import development_principal, get_current_principal
from app.identity.models import Permission, Role
from app.integrations.models import IntegratedAnalysisRequest
from app.integrations.orchestrator import IndustrialIntegrationOrchestrator
from app.llm.base import ModelGenerationCancelled
from app.main import app
from app.tools.base import Tool, ToolRisk
from app.tools.registry import ToolRegistry
from app.tools.simple_tools import CalculateTool, SummarizeNumbersTool, ConvertUnitsTool
from test_structured_plan import Script, PLAN, GOAL
from diagnostic_fixtures import payload as diagnostic_payload


@pytest.fixture
def storage(tmp_path, monkeypatch):
    from app.rag.embeddings import configured_embedding_provider
    configured_embedding_provider.cache_clear()
    monkeypatch.setenv("SOVEREIGN_AUTH_MODE", "disabled")
    monkeypatch.setenv("SOVEREIGN_EMBEDDING_PROVIDER", "hash")
    get_settings.cache_clear()
    settings = get_settings().model_copy(update={"workspace_root": tmp_path})
    monkeypatch.setattr(tasks, "get_settings", lambda: settings)
    engine = create_engine(f"sqlite:///{(tmp_path / 'checkpoints.db').as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine)
    # Direct handler tests run before application startup. Keep every imported
    # cache factory on this isolated schema, not an existing developer database.
    from app.resources import cache as cache_module
    from app.resources.cache import SQLiteCache
    monkeypatch.setattr(cache_module, "SQLiteCache", lambda: SQLiteCache(factory))
    monkeypatch.setattr(tasks, "SessionLocal", factory)
    monkeypatch.setattr(tasks, "task_event_broker", TaskEventBroker())
    def database():
        with factory() as db:
            yield db
    app.dependency_overrides[get_db] = database
    try:
        yield factory, tmp_path
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_principal, None)
        get_settings.cache_clear()
        configured_embedding_provider.cache_clear()
        engine.dispose()


def initial(db, *, request="Use calculate", run_id="durable-run"):
    payload = tasks.CreateTaskRequest(request=request, tool_scope="NUMERIC")
    state = tasks._accepted_state(payload, run_id)
    accept_task(db, state, payload.model_dump(mode="json"), development_principal())
    return state


def saved(db, run_id):
    return AgentRunState.model_validate_json(db.get(AgentRunRecord, run_id, populate_existing=True).state_json)


def events(client, task_id):
    return [json.loads(line[6:]) for line in client.get(f"/api/tasks/{task_id}/events").text.splitlines() if line.startswith("data: ")]


def test_acceptance_and_successful_results_are_committed_before_live_events(storage, monkeypatch):
    factory, _ = storage
    script = Script(PLAN)
    monkeypatch.setattr(tasks, "configured_local_provider", lambda *a, **k: SimpleNamespace(model="test", provider=script))
    observed = []
    async def inspect(kind, data):
        if kind == "tool_completed":
            with factory() as reader:
                row = reader.scalar(select(AgentRunRecord))
                state = saved(reader, row.id)
                assert state.status == RunStatus.running
                assert state.tool_records[-1]["execution_status"] == "succeeded"
                assert state.tool_records[-1]["output"]
                observed.append(state.tool_records[-1])
    with factory() as db:
        result = asyncio.run(tasks._execute_task(tasks.CreateTaskRequest(request=GOAL, tool_scope="NUMERIC"), db, development_principal(), inspect))
        assert result.status == RunStatus.completed and len(observed) == 3
        assert observed[0]["output"]["mean"] == 20
        assert observed[1]["arguments"] == {"expression": "20.0 * 8"}
        assert observed[2]["arguments"]["value"] == 160
        snapshots = db.scalars(select(TaskCheckpointRecord).where(TaskCheckpointRecord.run_id == result.id).order_by(TaskCheckpointRecord.revision)).all()
        assert [item.revision for item in snapshots] == list(range(1, len(snapshots) + 1))
        assert snapshots[0].boundary == "accepted" and snapshots[-1].phase == "finished"
        assert sum(item.boundary == "tool_started" for item in snapshots) == 3
        assert sum(item.boundary == "tool_result_saved" for item in snapshots) == 3
        assert all(sha256(item.state_json.encode()).hexdigest() == item.state_sha256 for item in snapshots)
        assert result.checkpoint.revision == snapshots[-1].revision
        assert not result.checkpoint.requires_reconciliation
        assert json.loads(db.get(TaskExecutionRecord, result.id).payload_json)["tool_scope"] == "NUMERIC"
    assert len(script.prompts) == 1


def test_cancel_between_steps_preserves_first_result_and_does_not_start_next_tool(storage, monkeypatch):
    factory, _ = storage
    monkeypatch.setattr(tasks, "configured_local_provider", lambda *a, **k: SimpleNamespace(model="test", provider=Script(PLAN)))
    async def run():
        cancelled = asyncio.Event()
        async def emit(kind, data):
            if kind == "tool_completed":
                cancelled.set()
        with factory() as db:
            with pytest.raises(ModelGenerationCancelled):
                await tasks._execute_task(tasks.CreateTaskRequest(request=GOAL, tool_scope="NUMERIC"), db, development_principal(), emit, cancelled)
            state = saved(db, db.scalar(select(AgentRunRecord.id)))
            assert state.status == RunStatus.cancelled
            assert len(state.tool_records) == 1 and state.tool_records[0]["output"]["mean"] == 20
            assert state.structured_plan["steps"][0]["status"] == "completed"
            assert state.structured_plan["steps"][1]["status"] == "pending"
            assert state.checkpoint.phase == "finished" and not state.checkpoint.requires_reconciliation
    asyncio.run(run())


def test_cancel_during_tool_keeps_previous_output_and_marks_active_outcome_unknown(storage, monkeypatch):
    factory, _ = storage
    async def run():
        started, cleaned, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        class Blocking(Tool):
            name, description, risk = "convert_units", "Wait for controlled interruption", ToolRisk.low
            input_schema = {"type": "object", "properties": {}, "additionalProperties": False}
            async def execute(self, args):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()
        tools = ToolRegistry()
        tools.register(CalculateTool()); tools.register(Blocking())
        plan = {"action": "plan", "reason_summary": "Calculate, then call the test tool.", "steps": [
            {"id": "first", "tool": "calculate", "arguments": {"expression": "2+3"}, "depends_on": []},
            {"id": "blocked", "tool": "convert_units", "arguments": {}, "depends_on": ["first"]}]}
        monkeypatch.setattr(tasks, "create_numeric_registry", lambda: tools)
        monkeypatch.setattr(tasks, "configured_local_provider", lambda *a, **k: SimpleNamespace(model="test", provider=Script(plan)))
        async def interrupt():
            await started.wait()
            cancelled.set()
        interrupt_task = asyncio.create_task(interrupt())
        with factory() as db:
            with pytest.raises(ModelGenerationCancelled):
                await tasks._execute_task(tasks.CreateTaskRequest(request="Use calculate then convert_units", tool_scope="NUMERIC"), db, development_principal(), cancellation_event=cancelled)
            await interrupt_task
            state = saved(db, db.scalar(select(AgentRunRecord.id)))
            assert cleaned.is_set() and state.status == RunStatus.cancelled
            assert state.tool_records[0]["output"]["result"] == 5
            assert state.tool_records[1]["execution_status"] == "unknown"
            assert state.tool_records[1]["success"] is None and "output" not in state.tool_records[1]
            assert state.checkpoint.requires_reconciliation
    asyncio.run(run())


def test_unhandled_callback_error_preserves_already_committed_tool_result(storage, monkeypatch):
    factory, _ = storage
    monkeypatch.setattr(tasks, "configured_local_provider", lambda *a, **k: SimpleNamespace(model="test", provider=Script(PLAN)))
    async def broken(kind, data):
        if kind == "tool_completed":
            raise RuntimeError("Test stream callback stopped")
    with factory() as db:
        with pytest.raises(RuntimeError, match="callback stopped"):
            asyncio.run(tasks._execute_task(tasks.CreateTaskRequest(request=GOAL, tool_scope="NUMERIC"), db, development_principal(), broken))
        state = saved(db, db.scalar(select(AgentRunRecord.id)))
        assert state.status == RunStatus.failed
        assert state.tool_records[0]["output"]["mean"] == 20
        assert state.checkpoint.phase == "finished" and not state.checkpoint.requires_reconciliation


def test_parallel_service_result_is_durable_while_other_service_is_still_pending(storage):
    factory, _ = storage
    async def run():
        graph_done = asyncio.Event()
        class Control:
            async def precheck(self, payload):
                return {"decision": {"action": "allow"}}
        class Graph:
            async def retrieve(self, *args):
                graph_done.set()
                return {"chunks": [{"id": "evidence-1", "content": "Recorded evidence"}]}
        class Diagnostics:
            async def diagnose(self, payload):
                await asyncio.Event().wait()
        with factory() as db:
            state = initial(db)
            writer = CheckpointWriter(db, state, development_principal()); writer.claim()
            with checkpoint_context(writer):
                task = asyncio.create_task(IndustrialIntegrationOrchestrator(
                    graph=Graph(), diagnostics=Diagnostics(), controlplane=Control(), provider=object(), model="test",
                ).analyze(IntegratedAnalysisRequest(query="Analyze this bearing", diagnostic=diagnostic_payload("bearing")), principal=development_principal()))
                await asyncio.wait_for(graph_done.wait(), 2)
                with factory() as reader:
                    snapshot = saved(reader, state.id)
                    assert snapshot.checkpoint.data["operation_outputs"]["service:graph-rag"]["chunks"][0]["id"] == "evidence-1"
                    assert "service:diagnostics" in snapshot.checkpoint.data["active_operations"]
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                writer.stop(RunStatus.cancelled, "Cancelled while diagnostics were pending.")
            assert saved(db, state.id).checkpoint.data["operation_outputs"]["service:graph-rag"]["chunks"]
    asyncio.run(run())


def test_failure_to_commit_before_a_tool_prevents_its_execution(storage, monkeypatch):
    factory, _ = storage
    invoked = []
    class Counted(SummarizeNumbersTool):
        async def execute(self, args):
            invoked.append(args)
            return await super().execute(args)
    tools = ToolRegistry()
    tools.register(Counted()); tools.register(CalculateTool()); tools.register(ConvertUnitsTool())
    original = CheckpointWriter.save
    def fail_before_tool(self, state, boundary, **kwargs):
        if boundary == "tool_started":
            raise OSError("Test checkpoint storage failed")
        return original(self, state, boundary, **kwargs)
    monkeypatch.setattr(CheckpointWriter, "save", fail_before_tool)
    monkeypatch.setattr(tasks, "create_numeric_registry", lambda: tools)
    monkeypatch.setattr(tasks, "configured_local_provider", lambda *a, **k: SimpleNamespace(model="test", provider=Script(PLAN)))
    with factory() as db:
        with pytest.raises(HTTPException, match="storage failed"):
            asyncio.run(tasks._execute_task(tasks.CreateTaskRequest(request=GOAL, tool_scope="NUMERIC"), db, development_principal()))
        assert invoked == []
        assert saved(db, db.scalar(select(AgentRunRecord.id))).status == RunStatus.failed


def test_expired_owner_is_fenced_and_live_leases_pauses_and_final_runs_are_preserved(storage):
    factory, _ = storage
    with factory() as db:
        state = initial(db)
        writer = CheckpointWriter(db, state, development_principal()); writer.claim()
        state.status = RunStatus.running
        state.tool_records = [{"tool": "calculate", "success": True, "execution_status": "succeeded", "output": {"result": 5}},
                              {"tool": "generate_docx", "success": None, "execution_status": "in_flight"}]
        writer.save(state, "tool_started")
        assert recover_expired_tasks(db) == 0
        lease = db.get(TaskExecutionRecord, state.id)
        lease.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()
    with factory() as restarted:
        assert recover_expired_tasks(restarted) == 1
        result = saved(restarted, state.id)
        assert result.status == RunStatus.interrupted and not result.lifecycle.terminal
        assert result.tool_records[0]["output"] == {"result": 5}
        assert result.tool_records[1]["execution_status"] == "unknown"
        assert result.checkpoint.requires_reconciliation
        assert recover_expired_tasks(restarted) == 0
    with pytest.raises(RuntimeError, match="lease is no longer owned"):
        writer.save(state, "stale_worker_write")
    with factory() as db:
        assert saved(db, state.id).status == RunStatus.interrupted


def test_pending_acceptance_survives_worker_never_starting(storage):
    factory, _ = storage
    with factory() as db:
        state = initial(db)
        execution = db.get(TaskExecutionRecord, state.id)
        execution.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()
        assert recover_expired_tasks(db, state.id) == 1
        restored = saved(db, state.id)
        assert restored.status == RunStatus.interrupted
        assert not restored.checkpoint.requires_reconciliation


def test_duplicate_worker_cannot_claim_the_same_task(storage):
    factory, _ = storage
    with factory() as db:
        state = initial(db)
        CheckpointWriter(db, state, development_principal()).claim()
    with factory() as db:
        with pytest.raises(RuntimeError, match="already claimed"):
            CheckpointWriter(db, state, development_principal()).claim()


def test_heartbeat_renews_lease_and_observes_cross_worker_cancellation(storage, monkeypatch):
    factory, _ = storage
    monkeypatch.setattr(checkpoints, "HEARTBEAT_SECONDS", .01)
    async def run():
        with factory() as db:
            state = initial(db)
            writer = CheckpointWriter(db, state, development_principal()); writer.claim()
            deadline = db.get(TaskExecutionRecord, state.id).lease_expires_at
            cancellation = asyncio.Event()
            heartbeat = asyncio.create_task(writer.heartbeat(cancellation))
            await asyncio.sleep(.04)
            with factory() as reader:
                execution = reader.get(TaskExecutionRecord, state.id)
                assert execution.lease_expires_at > deadline
                assert recover_expired_tasks(reader) == 0
                execution.cancel_requested = True
                reader.commit()
            await asyncio.wait_for(cancellation.wait(), 1)
            await heartbeat
    asyncio.run(run())


@pytest.mark.parametrize("reset_stream", [False, True])
def test_reconnect_uses_durable_results_without_starting_a_model_or_worker(storage, monkeypatch, reset_stream):
    factory, _ = storage
    with TestClient(app) as client:
        started = client.post("/api/tasks/start", json={"request": "Diagnose this bearing vibration anomaly"}).json()
        first = events(client, started["task_id"])
        result = first[-1]["payload"]["result"]
        assert result["id"] == started["task_id"]
        assert first[-1]["type"] == "task_paused"
        if reset_stream:
            monkeypatch.setattr(tasks, "task_event_broker", TaskEventBroker())
        async def forbidden(*args, **kwargs):
            raise AssertionError("Reconnecting must not execute anything")
        monkeypatch.setattr(tasks, "_execute_task", forbidden)
        replay = events(client, started["task_id"])
        assert replay[-1]["type"] == "task_paused"
        if reset_stream:
            assert replay[-1]["payload"]["replayed"]
        assert replay[-1]["payload"]["result"] == client.get(f"/api/tasks/{result['id']}").json()
        history = client.get(f"/api/tasks/{result['id']}/checkpoints").json()
        assert history[0]["phase"] == "paused" and history[-1]["boundary"] == "accepted"
        assert client.delete(f"/api/tasks/{result['id']}").json()["status"] == "cancelled"
        assert client.get(f"/api/tasks/{result['id']}").json()["status"] == "cancelled"
        assert events(client, result["id"])[-1]["type"] == "task_cancelled"


def test_history_and_snapshot_respect_existing_workspace_authorization(storage, monkeypatch):
    factory, _ = storage
    owner = development_principal().model_copy(update={"user_id": "owner", "organization_id": "org-a", "workspace_ids": ["plant-a"], "department_ids": ["maintenance"]})
    with factory() as db:
        payload = tasks.CreateTaskRequest(request="Explain pump cavitation")
        accept_task(db, tasks._accepted_state(payload, "private-run"), payload.model_dump(mode="json"), owner)
    outsider = owner.model_copy(update={"user_id": "outsider", "organization_id": "org-b", "workspace_ids": ["plant-b"], "roles": [Role.engineer], "permissions": [Permission.task_read]})
    settings = tasks.get_settings().model_copy(update={"auth_mode": "local"})
    monkeypatch.setattr(tasks, "get_settings", lambda: settings)
    app.dependency_overrides[get_current_principal] = lambda: outsider
    with TestClient(app) as client:
        assert client.get("/api/tasks/private-run").status_code == 404
        assert client.get("/api/tasks/private-run/checkpoints").status_code == 404
        assert client.get("/api/tasks").json() == []


def test_real_worker_process_exit_keeps_exact_results_and_does_not_replay_artifact(storage, monkeypatch):
    factory, root = storage
    child = r'''
import asyncio, os, sys
from pathlib import Path
from types import SimpleNamespace
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.api import tasks
from app.core.database import Base
from app.core.config import get_settings
from app.identity.dependencies import development_principal
from app.identity.authorization import AuthorizationService
from app.tools.registry import ToolRegistry
from app.tools.simple_tools import CalculateTool
from app.tools.application_tools import GenerateDocxTool
from app.artifacts.service import ArtifactService
from app.llm.base import StructuredGenerationResult
root = Path(sys.argv[1])
engine = create_engine('sqlite:///' + (root / 'checkpoints.db').as_posix())
Base.metadata.create_all(engine)
plan = {'action':'plan','reason_summary':'Calculate and create a report.','steps':[
 {'id':'calc','tool':'calculate','arguments':{'expression':'2+3'},'depends_on':[]},
 {'id':'report','tool':'generate_docx','arguments':{'filename':'crash.docx','title':'Verified result',
 'sections':[{'heading':'Calculation','content':'{{calc.result}}'}]},'depends_on':['calc']}]}
class Script:
 async def generate_json(self, *args, **kwargs):
  return StructuredGenerationResult(text='',model='test',provider='test',data=plan)
class CrashDocx(GenerateDocxTool):
 async def execute(self,args):
  result = await super().execute(args)
  assert result.success
  os._exit(0)  # Simulate process loss after the side effect, before the host receipt.
with Session(engine) as db:
 tools = ToolRegistry(); tools.register(CalculateTool())
 tools.register(CrashDocx(ArtifactService(db,root/'artifacts'), AuthorizationService.owned_scope(development_principal())))
 settings = get_settings().model_copy(update={'workspace_root':root})
 tasks.get_settings = lambda: settings
 tasks.create_agent_registry = lambda *args: tools
 tasks.configured_local_provider = lambda *args,**kwargs: SimpleNamespace(model='test',provider=Script())
 asyncio.run(tasks._execute_task(tasks.CreateTaskRequest(request='Use calculate then generate_docx to create a report.',chat_mode='CONTROLLED'),db,development_principal(),run_id='crashed-run'))
raise RuntimeError('Expected the child to exit inside artifact generation')
'''
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(Path(__file__).resolve().parents[2] / "backend"), SOVEREIGN_AUTH_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, "-c", child, str(root)], env=environment, capture_output=True, text=True, timeout=30,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    with factory() as db:
        before = saved(db, "crashed-run")
        assert before.status == RunStatus.running
        assert before.tool_records[0]["output"]["result"] == 5
        assert before.tool_records[1]["execution_status"] == "in_flight"
        artifact = db.scalar(select(ArtifactRecord))
        artifact_path = root / "artifacts" / artifact.path
        assert artifact.run_id == "crashed-run" and artifact_path.is_file()
        original_bytes = artifact_path.read_bytes()
        lease = db.get(TaskExecutionRecord, "crashed-run")
        lease.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()
    # Fresh application/session reads perform recovery, never inference or replay.
    with TestClient(app) as client:
        restored = client.get("/api/tasks/crashed-run").json()
        assert restored["status"] == "interrupted"
        assert restored["tool_records"][0]["output"]["result"] == 5
        assert restored["tool_records"][1]["execution_status"] == "unknown"
        assert restored["checkpoint"]["requires_reconciliation"]
        assert client.get("/api/tasks/crashed-run/checkpoints").json()[0]["boundary"] == "worker_interrupted"
        assert client.get("/api/tasks/crashed-run").json() == restored
        script = Script({"final_response": "The calculation and report completed.", "reason_summary": "Actual receipts verified."})
        monkeypatch.setattr(tasks, "configured_local_provider", lambda *a, **k: SimpleNamespace(model="test", provider=script))
        resumed = client.post("/api/tasks/crashed-run/resume", json={"checkpoint_revision": restored["checkpoint"]["revision"]})
        assert resumed.status_code == 202, resumed.text
        stream = events(client, "crashed-run")
        final = stream[-1]["payload"]["result"]
        assert final["status"] == "completed", final
        assert len(final["tool_records"]) == 2
        assert final["tool_records"][1]["reconciliation"]["type"] == "tool_receipt"
        assert not [item for item in stream if item["type"] == "tool_started"]
    with factory() as db:
        assert len(db.scalars(select(ArtifactRecord)).all()) == 1
        assert artifact_path.read_bytes() == original_bytes
