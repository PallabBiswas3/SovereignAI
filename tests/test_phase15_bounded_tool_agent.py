import asyncio
from pathlib import Path

from app.agent.state import AgentPlan, AgentRunState, RunStatus
from app.agent.tool_agent import BoundedToolAgent
from app.core.config import get_settings
from app.governance.action_guard import ActionGuard
from app.llm.base import GenerationResult, LocalModelProvider, StructuredGenerationResult
from app.router.schemas import RoutingDecision, TaskProfile
from app.tools.base import Tool, ToolResult, ToolRisk
from app.tools.registry import ToolRegistry


class EchoTool(Tool):
    name = "read_file"
    description = "Test registered read tool"
    risk = ToolRisk.low
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    async def execute(self, args):
        return ToolResult(success=True, output={"path": args["path"], "text": "verified pump evidence"})


class ScriptedProvider(LocalModelProvider):
    def __init__(self, decisions):
        self.decisions = iter(decisions)

    async def generate(self, prompt, model, system=None):
        return GenerationResult(text="", model=model, provider="test")

    async def generate_json(self, prompt, model, schema, system=None):
        data = next(self.decisions)
        return StructuredGenerationResult(text="", model=model, provider="test", data=data)

    async def health_check(self):
        return {"available": True}


def _state() -> AgentRunState:
    profile = TaskProfile(
        task_type="document", coding_requirement=0, reasoning_requirement=.5,
        vision_requirement=0, document_requirement=1, summarization_requirement=.5,
        latency_priority=.5, context_length_required=1024,
    )
    routing = RoutingDecision(
        selected_model="test", model_id="general", confidence=1, reason="test",
        task_profile=profile, scores={"general": 1},
    )
    return AgentRunState(
        id="bounded-test", request="Read the report", status=RunStatus.running,
        routing=routing, plan=AgentPlan(goal="Read the report", steps=[]),
    )


def test_registered_tool_is_called_then_agent_completes() -> None:
    tools = ToolRegistry()
    tools.register(EchoTool())
    provider = ScriptedProvider([
        {"action": "tool", "tool_name": "read_file", "arguments": {"path": "report.md"}, "reason_summary": "Read supplied evidence."},
        {"action": "complete", "reason_summary": "Evidence reviewed.", "final_response": "The pump evidence was reviewed."},
    ])
    result = asyncio.run(BoundedToolAgent(
        provider, "test", tools, ActionGuard(get_settings().tools_config), max_tool_calls=2,
    ).execute(_state(), ["report.md"]))

    assert result.status == RunStatus.completed
    assert result.tool_records[0]["tool"] == "read_file"
    assert result.tool_records[0]["success"] is True
    assert result.final_response == "The pump evidence was reviewed."


def test_unregistered_tool_never_executes_and_decision_limit_stops_agent() -> None:
    tools = ToolRegistry()
    tools.register(EchoTool())
    provider = ScriptedProvider([
        {"action": "tool", "tool_name": "execute_shell", "arguments": {"command": "whoami"}, "reason_summary": "invalid"},
        {"action": "tool", "tool_name": "execute_shell", "arguments": {}, "reason_summary": "invalid again"},
    ])
    result = asyncio.run(BoundedToolAgent(
        provider, "test", tools, ActionGuard(get_settings().tools_config), max_decisions=2,
    ).execute(_state(), []))

    assert result.status == RunStatus.failed
    assert result.tool_records == []
    assert any("Decision limit" in warning for warning in result.warnings)

