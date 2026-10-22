"""Verify smaller prompts and factual formatting preserve execution and review."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent.prompt_context import catalogue, compact_observations, plan_catalogue
from app.agent.result_formatter import verified_result
from app.agent.state import RunStatus
from app.agent.tool_agent import BoundedToolAgent
from app.api import tasks
from app.core.config import get_settings
from app.core.database import Base, get_db
from app.main import app
from app.tools.registry import create_default_registry, create_numeric_registry
from app.tools.application_tools import GenerateDocxTool
from app.artifacts.service import ArtifactService
from app.governance.action_guard import ActionGuard
from app.identity.dependencies import development_principal
from sqlalchemy.orm import Session
from app.orchestration.structured_plan import validate_value
from test_structured_plan import ANSWER, GOAL, PLAN, Script, executor, state
from test_timing import ollama_reply, provider


def test_actual_three_tool_chain_completes_with_one_model_call(tmp_path):
    script = Script(PLAN)
    result = asyncio.run(executor(tmp_path, script).execute(state(), []))
    assert result.status == RunStatus.completed and len(script.prompts) == 1
    assert result.runtime_metrics["response_generation"]["mode"] == "verified_formatter"
    assert "160.0" in result.final_response and "1.6 bar" in result.final_response
    assert len(result.tool_records) == 3 and len(result.sources) == 3
    assert result.runtime_metrics["prompt_compaction"][0]["visible_tools"] == 3


@pytest.mark.parametrize("query", ["Explain the results", "Assess whether the pump is safe", "Which value is unacceptable?",
    "Recommend maintenance", "Compare the values", "Give a diagnosis", "Summarize the findings", "Cite sources"])
def test_interpretation_and_evidence_requests_retain_answer_generation(tmp_path, query):
    script = Script(PLAN, ANSWER)
    result = asyncio.run(executor(tmp_path, script).execute(state(GOAL + " " + query), []))
    assert len(script.prompts) == 2
    assert result.runtime_metrics["response_generation"]["mode"] == "llm"


def test_formatter_limits_to_complete_supported_receipts():
    calculate = {"tool": "calculate", "success": True, "output": {"expression": "1+2", "result": 3}}
    assert verified_result("Report the result", [calculate]) == "calculate: 1+2 = 3."
    for observations in [[], [{**calculate, "success": False}], [{**calculate, "error": "denied"}],
        [{**calculate, "output": {"result": float("nan")}}],
        [{"tool": "knowledge_search", "success": True, "output": {"results": []}}],
        [{"tool": "read_file", "success": True, "output": {"text": "Important evidence"}}],
        [calculate, {"tool": "unknown", "success": True}],
        [{"tool": "generate_docx", "success": True, "output": {"artifact": {"name": "bad", "url": "https://outside.example"}}}]]:
        assert verified_result("Report the result", observations) is None


def test_file_listing_metadata_and_artifact_receipts_are_generic():
    listing = {"tool": "list_files", "success": True, "output": [f"file{i}.txt" for i in range(55)]}
    text = verified_result("List workspace files", [listing])
    assert "55" in text and "Only the first 50" in text and "file54" not in text
    assert verified_result("List files", [{**listing, "output": []}]) == "Workspace files (0):\n"
    metadata = {"tool": "get_file_metadata", "success": True, "output": {"name": "evidence.txt", "size": 19, "extension": ".txt"}}
    assert "19 bytes" in verified_result("Show file metadata", [metadata])
    artifact = {"tool": "generate_xlsx", "success": True, "output": {"artifact": {"name": "evidence.xlsx", "url": "/api/artifacts/abc-123"}}}
    assert "/api/artifacts/abc-123" in verified_result("Create a spreadsheet", [artifact])


def test_compact_observations_keep_evidence_and_actual_stored_data():
    observations = [{"tool": "calculate", "success": True, "output": {"result": 3, "validation": "verified", "results": ["duplicate evidence"]}},
        {"tool": "knowledge_search", "success": True, "output": {"results": [{"text": "Observed pressure", "source": {"file": "SOP.pdf", "page": 2, "source_id": "source-7"}}]}}]
    original = deepcopy(observations)
    compact = json.loads(compact_observations(observations))
    assert compact[0]["output"] == {"result": 3}
    assert compact[1] == observations[1] and observations == original
    numeric = [{"tool": "calculate", "output": {"result": 3, "results": [{"text": "Repeated numeric evidence", "source": {
        "source_id": "calculation-7", "file": "Local tool: calculate", "kind": "local_computation"}}]}}]
    shortened = compact_observations(numeric)
    assert "calculation-7" in shortened and "Repeated numeric evidence" not in shortened


def test_shortlist_has_fallback_and_respects_scoped_tools(tmp_path):
    tools = create_default_registry(tmp_path)
    visible = plan_catalogue("Use read_file then calculate", tools, [])
    assert {item["name"] for item in visible.discover()} == {"read_file", "calculate"}
    assert plan_catalogue("Read an attachment", tools, ["input.txt"]) is tools
    assert plan_catalogue("Read file then do the requested work", tools, []) is tools
    numeric = plan_catalogue("Use read_file then calculate", create_numeric_registry(), [])
    assert [item["name"] for item in numeric.discover()] == ["calculate"]
    assert '"minimum"' not in catalogue(numeric)  # Constraints remain in the authoritative generation schema.
    assert '"output":"expression,result"' in catalogue(numeric)
    assert len(catalogue(tools)) < len(json.dumps(tools.discover()))


def test_report_citations_reject_strings_and_allow_scoped_source_metadata():
    schema = GenerateDocxTool.input_schema
    with pytest.raises(ValueError):
        validate_value({"title": "Evidence", "sections": [], "sources": ["read.text"]}, schema)
    validate_value({"title": "Evidence", "sections": [], "sources": [{
        "file": "input.txt", "page": None, "section": None, "kind": "authorized_file"}]}, schema)


def test_fallback_cannot_claim_an_explicitly_requested_artifact_without_running_its_tool(tmp_path):
    (tmp_path / "input.txt").write_text("Observed evidence")
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            tools = create_default_registry(tmp_path)
            tools.register(GenerateDocxTool(ArtifactService(db, tmp_path / "artifacts")))
            script = Script({"action": "tool", "tool_name": "read_file", "arguments": {"path": "input.txt"}},
                            *[{"action": "complete", "final_response": "I created the artifact."}] * 3)
            result = asyncio.run(BoundedToolAgent(script, "test", tools, ActionGuard(get_settings().tools_config),
                principal=development_principal(), max_decisions=4).execute(state("read_file input.txt then generate_docx"), []))
            assert result.status == RunStatus.failed and result.artifacts == []
            assert "I created the artifact" not in result.final_response
            assert any("generate_docx" in warning for warning in result.warnings)
    finally:
        engine.dispose()


def test_api_retains_final_grounding_sse_and_saved_mode_with_one_timed_call(tmp_path, monkeypatch):
    monkeypatch.setenv("SOVEREIGN_AUTH_MODE", "disabled")
    monkeypatch.setenv("SOVEREIGN_EMBEDDING_PROVIDER", "hash")
    get_settings.cache_clear()
    engine = create_engine(f"sqlite:///{(tmp_path / 'optimized.db').as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine)
    monkeypatch.setattr(tasks, "SessionLocal", factory)
    calls, grounding = [], []
    original = tasks.GroundingChecker.evaluate
    def check(self, *args, **kwargs):
        grounding.append(True)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(tasks.GroundingChecker, "evaluate", check)
    def http_handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"models": [{"name": "test"}]})
        calls.append(request)
        return httpx.Response(200, json=ollama_reply(PLAN))
    monkeypatch.setattr(tasks, "configured_local_provider", lambda *a, **k: SimpleNamespace(
        model="test", provider=provider(httpx.AsyncClient(transport=httpx.MockTransport(http_handler)))))
    def database():
        with factory() as db:
            yield db
    app.dependency_overrides[get_db] = database
    try:
        with TestClient(app) as client:
            started = client.post("/api/tasks/start", json={"request": GOAL, "tool_scope": "NUMERIC"})
            events = client.get(f"/api/tasks/{started.json()['task_id']}/events").text
            data = [json.loads(line[6:]) for line in events.splitlines() if line.startswith("data: ")]
            final = next(e["payload"]["result"] for e in data if e["type"] == "task_completed")
            assert len(calls) == 1 and grounding == [True]
            assert final["runtime_metrics"]["timing"]["llm_call_count"] == 1
            assert final["runtime_metrics"]["response_generation"]["mode"] == "verified_formatter"
            assert final["governance"]["policy"] == "internal_assistant"
            assert client.get(f"/api/tasks/{final['id']}").json()["runtime_metrics"] == final["runtime_metrics"]
    finally:
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()
        get_settings.cache_clear()
