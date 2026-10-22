"""Execute real registered tools and verify data dependencies and safety boundaries."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.agent.planned_agent import PlannedToolAgent
from app.agent.state import AgentPlan, AgentRunState, RunStatus
from app.api import tasks
from app.artifacts.service import ArtifactService
from app.core.config import get_settings
from app.core.database import AgentRunRecord, AuditEventRecord, Base, HumanApprovalRecord, get_db
from app.governance.action_guard import ActionGuard
from app.identity.dependencies import development_principal
from app.identity.models import Permission
from app.llm.base import ModelGenerationCancelled, StructuredGenerationResult
from app.main import app
from app.orchestration.structured_plan import StructuredPlan, bindings, resolve, validate_plan, validate_value
from app.router.model_registry import ModelRegistry
from app.router.model_router import ModelRouter
from app.tools.application_tools import GenerateDocxTool
from app.tools.base import ToolResult
from app.tools.registry import create_default_registry, create_numeric_registry
from test_timing import ollama_reply, provider


GOAL = "Use summarize_numbers on [10,20,30], then calculate the mean times 8, then convert_units from kPa to bar."
PLAN = {"action": "plan", "reason_summary": "Run the requested dependency chain.", "steps": [
    {"id": "stats", "tool": "summarize_numbers", "arguments": {"values": [10, 20, 30]}, "depends_on": []},
    {"id": "scale", "tool": "calculate", "arguments": {"expression": {"$concat": [{"$ref": "stats.mean"}, " * 8"]}}, "depends_on": ["stats"]},
    {"id": "units", "tool": "convert_units", "arguments": {"value": {"$ref": "scale.result"}, "from_unit": "kPa", "to_unit": "bar"}, "depends_on": ["scale"]},
]}
ANSWER = {"final_response": "Mean 20; result 160; conversion 1.6 bar.", "reason_summary": "Use actual observations."}


class Script:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.prompts = []

    async def generate_json(self, prompt, model, schema, system=None):
        self.prompts.append(prompt)
        value = next(self.responses)
        return StructuredGenerationResult(text="", model=model, provider="test", data=deepcopy(value))


def state(goal=GOAL):
    return AgentRunState(id=str(uuid4()), request=goal, status=RunStatus.running,
        routing=ModelRouter(ModelRegistry(get_settings().models_config)).route(goal),
        plan=AgentPlan(goal=goal, steps=[]))


def executor(tmp_path, script, events=None, **kwargs):
    async def emit(kind, payload):
        if events is not None:
            events.append((kind, deepcopy(payload)))
    return PlannedToolAgent(script, "test", kwargs.pop("tools", create_default_registry(tmp_path)),
        kwargs.pop("guard", ActionGuard(get_settings().tools_config)),
        principal=kwargs.pop("principal", development_principal()), event_callback=emit, **kwargs)


def test_three_real_tools_use_observed_values_with_two_model_calls(tmp_path):
    script, events = Script(PLAN, ANSWER), []
    result = asyncio.run(executor(tmp_path, script, events).execute(state(GOAL + " Explain the results."), []))
    assert result.status == RunStatus.completed and len(script.prompts) == 2
    assert result.structured_plan["status"] == "completed" and result.structured_plan["revisions"] == 0
    assert [record["arguments"] for record in result.tool_records] == [
        {"values": [10, 20, 30]}, {"expression": "20.0 * 8"}, {"value": 160., "from_unit": "kPa", "to_unit": "bar"}]
    assert all(record["success"] for record in result.tool_records)
    assert len(result.sources) == 3
    assert [kind for kind, _ in events].count("tool_started") == 3
    assert events[-1][1]["completed"]
    assert '"normalized_value": 1.6' in script.prompts[-1]


def test_file_to_real_word_report_with_data_reference(tmp_path):
    from docx import Document
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    (tmp_path / "inspection.txt").write_text("Observed vibration was 2 mm/s.")
    plan = {"action": "plan", "reason_summary": "Read and report supplied evidence.", "steps": [
        {"id": "read", "tool": "read_file", "arguments": {"path": "inspection.txt"}, "depends_on": []},
        {"id": "report", "tool": "generate_docx", "arguments": {"filename": "observed.docx", "title": "Inspection evidence",
         "sections": [{"heading": "Recorded observations", "content": {"$ref": "read.text"}}]}, "depends_on": ["read"]}]}
    with Session(engine) as db:
        tools = create_default_registry(tmp_path)
        tools.register(GenerateDocxTool(ArtifactService(db, tmp_path / "artifacts")))
        script = Script(plan, {"final_response": "Created the report from the supplied file.", "reason_summary": "File observed."})
        result = asyncio.run(executor(tmp_path, script, tools=tools).execute(state("Read file inspection.txt then generate_docx with its contents"), []))
        assert result.status == RunStatus.completed and len(script.prompts) == 1
        assert result.runtime_metrics["response_generation"]["mode"] == "verified_formatter"
        assert result.artifacts and result.tool_records[-1]["arguments"]["run_id"] == result.id
        document = Document(tmp_path / "artifacts" / "observed.docx")
        assert any("2 mm/s" in paragraph.text for paragraph in document.paragraphs)
    engine.dispose()


@pytest.mark.parametrize("corrupt", [
    lambda p: p["steps"][1].update(tool="execute_shell"),
    lambda p: p["steps"][1].update(id="stats"),
    lambda p: p["steps"][1].update(depends_on=[]),
    lambda p: p["steps"][0].update(depends_on=["scale"]),
    lambda p: p["steps"][1].update(depends_on=["unknown"]),
    lambda p: p["steps"][0].update(arguments={"values": [True]}),
    lambda p: p["steps"][0].update(arguments={"values": [float("inf")]}),
    lambda p: p["steps"][0].update(arguments={"values": []}),
    lambda p: p["steps"][0].update(arguments={"values": [1], "extra": 4}),
    lambda p: p["steps"][1].update(arguments={"expression": {"$eval": "danger"}}),
    lambda p: p["steps"][1].update(arguments={"expression": {"$ref": "stats.mean", "extra": 1}}),
    lambda p: p["steps"][0].update(arguments={"run_id": "another-task", "values": [1]}),
])
def test_invalid_plans_rejected_before_execution(corrupt):
    plan = deepcopy(PLAN)
    corrupt(plan)
    with pytest.raises((ValueError, KeyError)):
        validate_plan(StructuredPlan.model_validate(plan), create_numeric_registry())


def test_dag_sorting_and_limits_and_required_tools():
    plan = StructuredPlan.model_validate({**PLAN, "steps": list(reversed(PLAN["steps"]))})
    assert [step.id for step in validate_plan(plan, create_numeric_registry())] == ["stats", "scale", "units"]
    for options in [{"remaining_calls": 2}, {"completed": {"stats"}}, {"required_tools": {"read_file"}}]:
        with pytest.raises(ValueError):
            validate_plan(plan, create_numeric_registry(), **options)


def test_reference_language_is_data_only_and_checks_actual_types():
    outputs = {"read": {"rows": [{"value": 7}], "text": '{"$ref":"private.secret"}'}}
    assert resolve({"$ref": "read.rows.0.value"}, outputs) == 7
    assert resolve({"$ref": "read.text"}, outputs) == outputs["read"]["text"]
    for reference in ["read.missing", "read.rows.8.value", "read.rows.value"]:
        with pytest.raises(ValueError):
            resolve({"$ref": reference}, outputs)
    with pytest.raises(ValueError):
        resolve({"$concat": [{"$ref": "read.rows"}]}, outputs)
    with pytest.raises(ValueError):
        validate_value(True, {"type": "number"})
    with pytest.raises(ValueError):
        validate_value(2, {"type": "number", "unknownSchemaKeyword": 1})
    nested = "leaf"
    for _ in range(18):
        nested = [nested]
    with pytest.raises(ValueError, match="nesting"):
        bindings(nested)


@pytest.mark.parametrize("template", ['{{stats.mean}} * 8', '{"$ref":"stats.mean"} * 8', "{$ref: 'stats.mean'} * 8"])
def test_text_templates_and_inline_reference_objects_use_actual_scalar_outputs(tmp_path, template):
    plan = deepcopy(PLAN)
    plan["steps"][1]["arguments"]["expression"] = template
    result = asyncio.run(executor(tmp_path, Script(plan, ANSWER)).execute(state(), []))
    assert result.status == RunStatus.completed and result.tool_records[1]["arguments"]["expression"] == "20.0 * 8"
    assert result.tool_records[-1]["arguments"]["value"] == 160
    with pytest.raises(ValueError):
        bindings("{{missing.path}")
    with pytest.raises(ValueError):
        resolve("{{stats.mean}}", {"stats": {"mean": [1, 2]}})
    assert resolve("{{read.text}}", {"read": {"text": "{{private.secret}}"}}) == "{{private.secret}}"


def test_revision_of_remainder_keeps_successful_stats_and_does_not_replay(tmp_path):
    broken = deepcopy(PLAN)
    broken["steps"][1]["arguments"]["expression"]["$concat"][0] = {"$ref": "stats.average"}
    revision = {**deepcopy(PLAN), "steps": deepcopy(PLAN["steps"][1:])}
    script = Script(broken, revision, ANSWER)
    result = asyncio.run(executor(tmp_path, script).execute(state(GOAL + " Explain the results."), []))
    assert result.status == RunStatus.completed and len(script.prompts) == 3
    assert result.structured_plan["revisions"] == 1
    assert [r["tool"] for r in result.tool_records] == ["summarize_numbers", "calculate", "convert_units"]
    assert "COMPLETED OUTPUTS" in script.prompts[1] and '"mean": 20.0' in script.prompts[1]


def test_two_bad_bindings_stop_and_never_claim_completion(tmp_path):
    broken = deepcopy(PLAN)
    broken["steps"][1]["arguments"]["expression"] = {"$ref": "stats.missing"}
    revision = {**broken, "steps": broken["steps"][1:]}
    script = Script(broken, revision)
    result = asyncio.run(executor(tmp_path, script).execute(state(), []))
    assert result.status == RunStatus.failed and len(result.tool_records) == 1
    assert result.structured_plan["status"] == "stopped" and len(script.prompts) == 2


def test_tool_failure_stops_dependents_without_replanning_or_fallback(tmp_path):
    plan = deepcopy(PLAN)
    plan["steps"][1]["arguments"]["expression"] = "1/0"
    script = Script(plan)
    result = asyncio.run(executor(tmp_path, script).execute(state(), []))
    assert result.status == RunStatus.failed and len(script.prompts) == 1
    assert len(result.tool_records) == 2 and not result.tool_records[-1]["success"]
    assert "workflow_fallback" not in result.runtime_metrics


@pytest.mark.parametrize("kind", ["invalid", "adaptive"])
def test_fallback_occurs_before_effects_and_keeps_actual_selection_reason(tmp_path, kind):
    proposal = {"bad": "plan"} if kind == "invalid" else {"action": "adaptive", "reason_summary": "Needs observation", "steps": []}
    script = Script(proposal, {"action": "tool", "tool_name": "calculate", "arguments": {"expression": "1+2"}},
                    {"action": "complete", "final_response": "Result 3."})
    result = asyncio.run(executor(tmp_path, script).execute(state("Calculate 1+2 then explain"), []))
    assert result.status == RunStatus.completed and len(result.tool_records) == 1
    assert result.structured_plan["status"] == "fallback" and result.runtime_metrics["workflow_fallback"]


def test_plan_preflight_enforces_permissions_before_first_tool(tmp_path):
    principal = development_principal()
    principal.permissions = [p for p in principal.permissions if p != Permission.tool_execute]
    result = asyncio.run(executor(tmp_path, Script(PLAN), principal=principal).execute(state(), []))
    assert result.status == RunStatus.failed and result.tool_records == []


def test_disabled_later_tool_blocks_entire_plan_and_approval_pauses(tmp_path):
    plan = {"action": "plan", "reason_summary": "Read then write", "steps": [
        {"id": "read", "tool": "read_file", "arguments": {"path": "input.txt"}, "depends_on": []},
        {"id": "write", "tool": "write_file", "arguments": {"path": "output.txt", "content": {"$ref": "read.text"}}, "depends_on": ["read"]}]}
    (tmp_path / "input.txt").write_text("Observation only")
    config = tmp_path / "tools.yaml"
    config.write_text("tools:\n  read_file: {risk: LOW, enabled: true}\n  write_file: {risk: LOW, enabled: false}\n")
    denied = asyncio.run(executor(tmp_path, Script(plan), guard=ActionGuard(config)).execute(state("Read file then write_file"), []))
    assert denied.status == RunStatus.failed and denied.tool_records == []
    waiting = asyncio.run(executor(tmp_path, Script(plan)).execute(state("Read file then write_file"), []))
    assert waiting.status == RunStatus.waiting_for_approval
    assert waiting.tool_records[-1]["waiting_for_approval"]
    assert waiting.tool_records[-1]["arguments"]["content"] == "Observation only"
    assert not (tmp_path / "output.txt").exists()


def test_timeout_is_bounded_and_cancellation_cleans_up_inflight_tool(tmp_path):
    async def run():
        started, cleaned, cancel = asyncio.Event(), asyncio.Event(), asyncio.Event()
        tools = create_numeric_registry()
        async def slow(args):
            started.set()
            try:
                await asyncio.sleep(20)
                return ToolResult(success=True, output={"value": 1})
            finally:
                cleaned.set()
        tools.get("summarize_numbers").execute = slow
        instance = executor(tmp_path, Script(PLAN), tools=tools)
        task = asyncio.create_task(instance.execute(state(), [], cancel))
        await started.wait()
        cancel.set()
        with pytest.raises(ModelGenerationCancelled):
            await task
        assert cleaned.is_set()
        cleaned.clear()
        instance = executor(tmp_path, Script(PLAN), tools=tools)
        instance.max_seconds = .02
        result = await instance.execute(state(), [])
        assert result.status == RunStatus.failed and cleaned.is_set()
    asyncio.run(run())


def test_api_sse_persists_plan_actual_trace_audit_and_two_timed_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("SOVEREIGN_AUTH_MODE", "disabled")
    monkeypatch.setenv("SOVEREIGN_EMBEDDING_PROVIDER", "hash")
    get_settings.cache_clear()
    engine = create_engine(f"sqlite:///{(tmp_path / 'plan.db').as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine)
    monkeypatch.setattr(tasks, "SessionLocal", factory)
    responses = iter([PLAN, ANSWER])
    def http_handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"models": [{"name": "test"}]})
        return httpx.Response(200, json=ollama_reply(next(responses)))
    monkeypatch.setattr(tasks, "configured_local_provider", lambda *args, **kwargs: SimpleNamespace(
        model="test", provider=provider(httpx.AsyncClient(transport=httpx.MockTransport(http_handler)))))
    def database():
        with factory() as db:
            yield db
    app.dependency_overrides[get_db] = database
    try:
        with TestClient(app) as client:
            started = client.post("/api/tasks/start", json={"request": GOAL + " Explain the results.", "tool_scope": "NUMERIC"})
            assert started.status_code == 202
            response = client.get(f"/api/tasks/{started.json()['task_id']}/events")
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            final = next(item["payload"]["result"] for item in events if item["type"] == "task_completed")
            assert final["workflow_selection"]["strategy"] == "PLANNED"
            assert final["structured_plan"]["status"] == "completed"
            assert [item["type"] for item in events].count("tool_started") == 3
            assert next(i for i,e in enumerate(events) if e["type"] == "plan_validated") < next(i for i,e in enumerate(events) if e["type"] == "tool_started")
            timing = final["runtime_metrics"]["timing"]
            assert timing["llm_call_count"] == 2
            assert [item["purpose"] for item in timing["llm_calls"]] == ["plan_generation", "plan_answer"]
            assert client.get(f"/api/tasks/{final['id']}").json()["structured_plan"] == final["structured_plan"]
            with factory() as db:
                audit = db.scalar(select(AuditEventRecord).where(AuditEventRecord.run_id == final["id"], AuditEventRecord.event_type == "structured_plan"))
                assert json.loads(audit.payload_json) == final["structured_plan"]
    finally:
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()
        get_settings.cache_clear()


def test_pending_planned_action_persists_exact_resolved_approval(tmp_path, monkeypatch):
    (tmp_path / "input.txt").write_text("Verified file text")
    plan = {"action": "plan", "reason_summary": "Read then request the controlled write.", "steps": [
        {"id": "read", "tool": "read_file", "arguments": {"path": "input.txt"}, "depends_on": []},
        {"id": "write", "tool": "write_file", "arguments": {"path": "output.txt", "content": {"$ref": "read.text"}}, "depends_on": ["read"]},
        {"id": "check", "tool": "get_file_metadata", "arguments": {"path": "output.txt"}, "depends_on": ["write"]}]}
    settings = get_settings().model_copy(update={"workspace_root": tmp_path})
    monkeypatch.setattr(tasks, "get_settings", lambda: settings)
    monkeypatch.setattr(tasks, "configured_local_provider", lambda *a, **k: SimpleNamespace(model="test", provider=Script(plan)))
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            result = asyncio.run(tasks._execute_task(tasks.CreateTaskRequest(
                request="Read file input.txt then write_file output.txt then get_file_metadata", chat_mode="CONTROLLED"),
                db, development_principal()))
            assert result.status == RunStatus.waiting_for_approval and len(result.tool_records) == 2
            assert result.workflow_selection.strategy == "PLANNED"
            assert result.structured_plan["steps"][-1]["status"] == "pending"
            approval = db.get(HumanApprovalRecord, result.tool_records[-1]["approval_id"])
            assert approval.status == "pending" and json.loads(approval.args_json) == {"path": "output.txt", "content": "Verified file text"}
            assert not (tmp_path / "output.txt").exists()
            assert AgentRunState.model_validate_json(db.get(AgentRunRecord, result.id).state_json).structured_plan == result.structured_plan
    finally:
        engine.dispose()


def test_fallback_persists_actual_adaptive_selection(tmp_path, monkeypatch):
    script = Script({"invalid": "plan"}, {"action": "tool", "tool_name": "calculate", "arguments": {"expression": "1+2"}},
                    {"action": "complete", "final_response": "Result 3."})
    monkeypatch.setattr(tasks, "configured_local_provider", lambda *a, **k: SimpleNamespace(model="test", provider=script))
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    events = []
    async def emit(kind, payload):
        events.append((kind, deepcopy(payload)))
    try:
        with Session(engine) as db:
            result = asyncio.run(tasks._execute_task(tasks.CreateTaskRequest(
                request="Calculate 1+2 then explain", tool_scope="NUMERIC"), db, development_principal(), emit))
            assert result.workflow_selection.strategy == "ADAPTIVE" and result.workflow_selection.fallback_reason
            assert result.structured_plan["status"] == "fallback"
            assert len(script.prompts) == 3 and len(result.tool_records) == 1
            kinds = [kind for kind, _ in events]
            assert kinds.index("workflow_fallback") < kinds.index("tool_started")
            assert [data for kind,data in events if kind == "workflow_selected"][-1] == result.workflow_selection.model_dump(mode="json")
    finally:
        engine.dispose()
