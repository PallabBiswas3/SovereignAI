from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent.state import AgentPlan, AgentRunState, RunStatus
from app.agent.tool_agent import BoundedToolAgent
from app.api import tasks
from app.core.config import get_settings
from app.core.database import Base, get_db
from app.governance.action_guard import ActionGuard
from app.identity.dependencies import development_principal, get_current_principal
from app.identity.models import Permission, Role
from app.llm.base import StructuredGenerationResult
from app.main import app
from app.orchestration.chat_mode import ChatMode, ChatModeSelector
from app.router.model_registry import ModelRegistry
from app.router.model_router import ModelRouter
from app.tools.registry import create_default_registry, create_numeric_registry
from app.tools.simple_tools import CalculateTool, ConvertUnitsTool, SummarizeNumbersTool


@pytest.mark.parametrize("expression,expected", [("(125 * 8) + 50", 1050), ("(30 - 10) / 2", 10),
                                                ("-3 + 2**3", 5), ("9 // 2 + 9 % 2", 5)])
def test_calculator(expression, expected):
    result = asyncio.run(CalculateTool().execute({"expression": expression}))
    assert result.success and result.output["result"] == expected
    assert result.output["results"][0]["source"]["basis"] == "user_supplied_inputs"


@pytest.mark.parametrize("expression", ["__import__('os').system('whoami')", "().__class__", "[1] * 99999999",
                                        "2 ** 9999999", "1/0", "1e999", "True + 2", "'4'", "9**9**9",
                                        "(1+" * 100, "1+" * 40 + "1", "(-1)**0.5"])
def test_calculator_rejects_code_and_resource_exhaustion(expression):
    result = asyncio.run(CalculateTool().execute({"expression": expression}))
    assert not result.success and result.output is None


@pytest.mark.parametrize("args", [{}, {"expression": 1}, {"expression": "1+2", "command": "anything"}])
def test_calculator_strict_arguments(args):
    assert not asyncio.run(CalculateTool().execute(args)).success


def test_number_summary_and_evidence():
    result = asyncio.run(SummarizeNumbersTool().execute({"values": [10, 20, 30]}))
    assert result.success
    assert {key: result.output[key] for key in ("count", "total", "minimum", "maximum", "mean", "median")} == {
        "count": 3, "total": 60, "minimum": 10, "maximum": 30, "mean": 20, "median": 20,
    }


def test_numeric_catalogue_contains_only_the_three_read_only_tools():
    assert {tool["name"] for tool in create_numeric_registry().discover()} == {"calculate", "summarize_numbers", "convert_units"}
    assert all(tool["risk"] == "LOW" for tool in create_numeric_registry().discover())


@pytest.mark.parametrize("extra", [{"attachments": ["uploads/file.csv"]}, {"workcell_id": "pump"},
                                  {"chat_mode": "GENERAL"}, {"chat_mode": "AUTHORIZED"}, {"tool_scope": "SHELL"}])
def test_numeric_scope_cannot_be_mixed_with_other_execution_paths(extra):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        tasks.CreateTaskRequest.model_validate({"request": GOAL, "tool_scope": "NUMERIC", **extra})


@pytest.mark.parametrize("values", [[], [True], ["20"], [float("nan")], [float("inf")], [1e20], list(range(257))])
def test_summary_bounds(values):
    assert not asyncio.run(SummarizeNumbersTool().execute({"values": values})).success


@pytest.mark.parametrize("value,source,target,expected", [(250, "kPa", "bar", 2.5), (160, "kPa", "bar", 1.6),
                                                       (1000, "mm", "m", 1), (0, "degC", "K", 273.15)])
def test_unit_conversion(value, source, target, expected):
    result = asyncio.run(ConvertUnitsTool().execute({"value": value, "from_unit": source, "to_unit": target}))
    assert result.success and result.output["normalized_value"] == pytest.approx(expected)


@pytest.mark.parametrize("args", [{"value": 1, "from_unit": "bar", "to_unit": "mm"},
                                 {"value": float("inf"), "from_unit": "kPa", "to_unit": "bar"},
                                 {"value": 1, "from_unit": "__import__('os')", "to_unit": "bar"},
                                 {"value": "250", "from_unit": "kPa", "to_unit": "bar"}])
def test_conversion_rejects_ambiguous_or_incompatible_input(args):
    assert not asyncio.run(ConvertUnitsTool().execute(args)).success


GOAL = "Use summarize_numbers on [10,20,30], then calculate the mean times 8, then convert_units from kPa to bar."


def state(goal=GOAL):
    routing = ModelRouter(ModelRegistry(get_settings().models_config)).route(goal)
    return AgentRunState(id=str(uuid4()), request=goal, status=RunStatus.running, routing=routing,
                         plan=AgentPlan(goal=goal, steps=[]))


class ScriptedProvider:
    def __init__(self, decisions=None):
        self.calls = 0
        self.decisions = iter(decisions) if decisions is not None else None

    async def generate_json(self, prompt, model, schema, system=None):
        self.calls += 1
        if self.decisions is not None:
            data = next(self.decisions)
        elif self.calls == 1:
            data = {"action": "tool", "tool_name": "summarize_numbers", "arguments": {"values": [10, 20, 30]}}
        elif self.calls == 2:
            assert '"mean": 20.0' in prompt
            data = {"action": "tool", "tool_name": "calculate", "arguments": {"expression": "20*8"}}
        elif self.calls == 3:
            assert '"result": 160.0' in prompt
            data = {"action": "tool", "tool_name": "convert_units", "arguments": {"value": 160, "from_unit": "kPa", "to_unit": "bar"}}
        else:
            assert '"normalized_value": 1.6' in prompt
            data = {"action": "complete", "final_response": "Mean 20; scaled result 160; converted value 1.6 bar."}
        return StructuredGenerationResult(text="", data={"reason_summary": "Use the observed result for the next step.", **data}, model=model, provider="test")


def agent(tmp_path, provider, events, principal=None, tools_config=None):
    async def emit(kind, payload):
        events.append((kind, payload))

    return BoundedToolAgent(provider, "test", create_default_registry(tmp_path),
        ActionGuard(tools_config or get_settings().tools_config), principal=principal or development_principal(),
        event_callback=emit, max_decisions=4)


def test_full_agent_uses_real_tools_observations_evidence_and_live_events(tmp_path):
    events = []
    result = asyncio.run(agent(tmp_path, ScriptedProvider(), events).execute(state(), []))
    assert result.status == RunStatus.completed
    assert [record["tool"] for record in result.tool_records] == ["summarize_numbers", "calculate", "convert_units"]
    assert all(record["success"] for record in result.tool_records)
    assert len(result.sources) == 3 and all(source["kind"] == "local_computation" for source in result.sources)
    assert [kind for kind, _ in events] == ["plan_created"] + ["tool_proposed", "tool_started", "tool_completed"] * 3 + ["agent_verified"]
    assert events[-1][1]["completed"] and not events[-1][1]["missing_tools"]
    assert all(step.verification for step in result.plan.steps if step.action in {"calculate", "convert_units", "summarize_numbers"})


def test_model_cannot_claim_named_tools_ran_without_executing_them(tmp_path):
    events = []
    provider = ScriptedProvider([{"action": "complete", "final_response": "I ran all tools."}] * 4)
    result = asyncio.run(agent(tmp_path, provider, events).execute(state(), []))
    assert result.status == RunStatus.failed and result.tool_records == []
    assert not events[-1][1]["completed"] and len(events[-1][1]["missing_tools"]) == 3


def test_partial_workflow_is_not_reported_as_complete(tmp_path):
    provider = ScriptedProvider([
        {"action": "tool", "tool_name": "calculate", "arguments": {"expression": "20*8"}},
        *[{"action": "complete", "final_response": "All done"}] * 3,
    ])
    result = asyncio.run(agent(tmp_path, provider, []).execute(state(), []))
    assert result.status == RunStatus.failed and len(result.tool_records) == 1
    assert any("did not complete" in warning for warning in result.warnings)


def test_authorization_denial_never_emits_tool_started(tmp_path):
    principal = development_principal().model_copy(update={"roles": [Role.user], "permissions": [Permission.task_create]})
    provider = ScriptedProvider([{"action": "tool", "tool_name": "calculate", "arguments": {"expression": "1+2"}}] * 4)
    events = []
    result = asyncio.run(agent(tmp_path, provider, events, principal).execute(state("Use calculate on 1+2"), []))
    assert result.status == RunStatus.failed
    assert all(not record["success"] for record in result.tool_records)
    assert not any(kind == "tool_started" for kind, _ in events)


def test_disabled_tool_and_approval_policy_are_preserved(tmp_path):
    config = tmp_path / "tools.yaml"
    config.write_text("tools:\n  calculate: {risk: HIGH, enabled: true, approval_required: true}\n")
    provider = ScriptedProvider([{"action": "tool", "tool_name": "calculate", "arguments": {"expression": "1+2"}}])
    events = []
    result = asyncio.run(agent(tmp_path, provider, events, tools_config=config).execute(state("Use calculate on 1+2"), []))
    assert result.status == RunStatus.waiting_for_approval
    assert not any(kind == "tool_started" for kind, _ in events)
    assert any(kind == "tool_approval_required" for kind, _ in events)


@pytest.mark.parametrize("prompt", [GOAL, "Calculate (125 * 8) + 50", "convert units from 250 kPa to bar", "summarize numbers 10,20,30"])
def test_simple_prompts_choose_existing_controlled_tool_path(prompt):
    assert ChatModeSelector().select(ChatMode.automatic, prompt).selected == ChatMode.controlled
    assert tasks._requires_tool_agent(tasks.CreateTaskRequest(request=prompt))


def test_task_api_and_sse_use_existing_governed_path_without_replayed_tool_events(tmp_path, monkeypatch):
    monkeypatch.setenv("SOVEREIGN_AUTH_MODE", "disabled")
    get_settings.cache_clear()
    engine = create_engine(f"sqlite:///{(tmp_path / 'agent.db').as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine)
    monkeypatch.setattr(tasks, "SessionLocal", factory)
    monkeypatch.setattr(tasks, "configured_local_provider", lambda *args, **kwargs: SimpleNamespace(provider=ScriptedProvider(), model="test"))

    def database():
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = database
    try:
        with TestClient(app) as client:
            started = client.post("/api/tasks/start", json={"request": "Investigate the example. " + GOAL, "chat_mode": "CONTROLLED", "tool_scope": "NUMERIC"})
            assert started.status_code == 202
            response = client.get(f"/api/tasks/{started.json()['task_id']}/events")
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            assert [event["type"] for event in events].count("tool_started") == 3
            assert [event["type"] for event in events].count("tool_completed") == 3
            final = next(event["payload"]["result"] for event in events if event["type"] == "task_completed")
            assert final["chat_mode"] == "CONTROLLED" and len(final["tool_records"]) == 3
            assert final["principal_id"] == "development-user" and final["governance"]["policy"] == "internal_assistant"
    finally:
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()
        get_settings.cache_clear()
