"""Deterministic timing-contract tests; mock durations are not performance evidence."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api import tasks
from app.core.config import get_settings
from app.core.database import Base, get_db
from app.llm.base import GenerationChunk, ModelGenerationCancelled, StructuredGenerationResult
from app.llm.ollama_provider import OllamaProvider
from app.llm.vllm_provider import VLLMProvider
from app.main import app
from app.monitoring.timing import capture_task_timing, measure_generation, measure_stream, timing_snapshot
from app.resources.lifecycle import ModelLifecycleManager
from app.resources.scheduler import ResourceScheduler


STATS = {"prompt_token_count": 120, "token_count": 30, "prompt_processing_seconds": .25,
         "generation_seconds": 1.5, "load_duration_seconds": .1, "queue_wait_seconds": 0,
         "tokens_per_second": 20, "warm_status": "warm"}


def provider(client, cls=OllamaProvider, **kwargs):
    return cls("http://127.0.0.1:11434", client=client, scheduler=ResourceScheduler(),
               lifecycle=ModelLifecycleManager(), **kwargs)


def ollama_reply(data):
    return {"model": "test", "response": json.dumps(data), "done": True,
            "eval_count": 30, "prompt_eval_count": 120, "total_duration": 1_850_000_000,
            "prompt_eval_duration": 250_000_000, "eval_duration": 1_500_000_000,
            "load_duration": 100_000_000}


@pytest.mark.parametrize("cls", [OllamaProvider, VLLMProvider])
def test_real_provider_adapter_records_each_structured_call_and_missing_measurements(cls):
    async def run():
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"models": [{"name": "test"}], "data": []})
            if cls is OllamaProvider:
                return httpx.Response(200, json=ollama_reply({"answer": "secret output"}))
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer":"secret output"}'}}],
                                            "usage": {"prompt_tokens": 120, "completion_tokens": 30}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            local = provider(client, cls)
            with capture_task_timing() as timing:
                for _ in range(2):
                    result = await local.generate_json("secret prompt", "test", {"type": "object"})
                    assert result.data == {"answer": "secret output"}
                report = timing.snapshot()
        assert report["llm_call_count"] == 2
        call = report["llm_calls"][0]
        assert call["prompt_token_count"] == 120 and call["token_count"] == 30
        assert call["outcome"] == "completed" and call["wall_seconds"] >= 0
        assert call["time_to_first_token_seconds"] is None
        assert "secret" not in json.dumps(report)
        if cls is OllamaProvider:
            assert call["prompt_processing_seconds"] == .25
            assert call["generation_seconds"] == 1.5
            assert call["load_duration_seconds"] == .1
            assert call["tokens_per_second"] == 20
        else:
            assert call["prompt_processing_seconds"] is None
            assert call["generation_seconds"] is None
            assert call["load_duration_seconds"] is None
        assert timing_snapshot() is None
    asyncio.run(run())


@pytest.mark.parametrize("cls", [OllamaProvider, VLLMProvider])
def test_streaming_generate_is_counted_once_and_preserves_tokens(cls):
    async def run():
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"models": [], "data": []})
            if cls is OllamaProvider:
                lines = [json.dumps({"response": "hello", "done": False}), json.dumps(ollama_reply({}))]
                return httpx.Response(200, text="\n".join(lines))
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                                  'data: {"choices":[],"usage":{"prompt_tokens":120,"completion_tokens":30}}\n\n'
                                  'data: [DONE]\n\n')
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with capture_task_timing() as timing:
                result = await provider(client, cls).generate("prompt", "test")
            assert result.text.startswith("hello")
            assert timing.snapshot()["llm_call_count"] == 1
            assert timing.llm_calls[0]["purpose"] == "streamed_generation"
            assert timing.llm_calls[0]["token_count"] == 30
    asyncio.run(run())


@pytest.mark.parametrize("cls", [OllamaProvider, VLLMProvider])
def test_missing_provider_token_counts_remain_unknown(cls):
    if cls is OllamaProvider:
        stats = cls._runtime_stats({}, started=0, first_token_at=None, was_loaded=None, queue_wait_seconds=0)
    else:
        stats = cls._runtime_stats({}, 0, None, 0, "test", None)
    assert stats["token_count"] is None and stats["prompt_token_count"] is None


@pytest.mark.parametrize("fallback", [True, False])
def test_provider_failure_is_recorded_without_changing_failure_behavior(fallback):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503))) as client:
            with capture_task_timing() as timing:
                local = provider(client, allow_fallback=fallback)
                if fallback:
                    assert (await local.generate_json("secret", "test", {})).fallback
                else:
                    with pytest.raises(RuntimeError):
                        await local.generate_json("secret", "test", {})
            assert timing.llm_calls[0]["outcome"] == ("fallback" if fallback else "failed")
            assert timing.llm_calls[0]["token_count"] is None
    asyncio.run(run())


def test_cancellation_and_concurrent_requests_have_isolated_metrics():
    class Local:
        @measure_generation
        async def generate_json(self, prompt, model):
            await asyncio.sleep(0)
            if model == "cancel":
                raise asyncio.CancelledError()
            return StructuredGenerationResult(text="", model=model, provider="test", data={}, runtime_stats=STATS)

    async def run():
        async def request(model):
            with capture_task_timing() as timing:
                try:
                    await asyncio.wait_for(Local().generate_json("private", model), timeout=1)
                except asyncio.CancelledError:
                    assert model == "cancel"
                return timing.snapshot()
        reports = await asyncio.gather(request("first"), request("second"), request("cancel"))
        for model, report in zip(["first", "second", "cancel"], reports):
            assert report["llm_call_count"] == 1
            assert report["llm_calls"][0]["model"] == model
        assert reports[2]["llm_calls"][0]["outcome"] == "cancelled"
        assert timing_snapshot() is None
    asyncio.run(run())


def test_closing_instrumented_stream_closes_underlying_generator():
    closed = []
    class Local:
        @measure_stream
        async def stream(self, prompt, model):
            try:
                yield GenerationChunk(text="first", model=model, provider="test")
                await asyncio.sleep(10)
            finally:
                closed.append(True)

    async def run():
        with capture_task_timing() as timing:
            stream = Local().stream("private", "test")
            await anext(stream)
            await stream.aclose()
        assert closed == [True]
        assert timing.llm_calls[0]["outcome"] == "cancelled"
    asyncio.run(run())


@pytest.mark.parametrize("error", [RuntimeError("failed"), ModelGenerationCancelled("cancelled")])
@pytest.mark.parametrize("broken_callback", [False, True])
def test_failed_or_cancelled_task_emits_partial_timing_and_preserves_original_error(monkeypatch, error, broken_callback):
    events = []
    async def execute(*args):
        raise error
    async def emit(kind, payload):
        events.append((kind, payload))
        if broken_callback:
            raise ValueError("Cannot save event")
    monkeypatch.setattr(tasks, "_execute_task_measured", execute)
    with pytest.raises(type(error), match=str(error)):
        asyncio.run(tasks._execute_task(None, None, None, emit))
    assert events[0][0] == "task_timing"
    assert events[0][1]["timing"]["partial"] is True
    assert "failed" not in json.dumps(events)
    assert timing_snapshot() is None


def test_api_sse_and_saved_run_retain_four_calls_and_three_actual_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("SOVEREIGN_AUTH_MODE", "disabled")
    monkeypatch.setenv("SOVEREIGN_EMBEDDING_PROVIDER", "hash")
    get_settings.cache_clear()
    engine = create_engine(f"sqlite:///{(tmp_path / 'timing.db').as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine)
    monkeypatch.setattr(tasks, "SessionLocal", factory)
    decisions = iter([
        {"action": "tool", "tool_name": "summarize_numbers", "arguments": {"values": [10, 20, 30]}},
        {"action": "tool", "tool_name": "calculate", "arguments": {"expression": "20*8"}},
        {"action": "tool", "tool_name": "convert_units", "arguments": {"value": 160, "from_unit": "kPa", "to_unit": "bar"}},
        {"action": "complete", "final_response": "Mean 20; result 160; conversion 1.6 bar."},
    ])
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"models": [{"name": "test"}]})
        return httpx.Response(200, json=ollama_reply(next(decisions)))

    def runtime(*args, **kwargs):
        return SimpleNamespace(model="test", provider=provider(httpx.AsyncClient(transport=httpx.MockTransport(handler))))

    monkeypatch.setattr(tasks, "configured_local_provider", runtime)
    def database():
        with factory() as db:
            yield db
    app.dependency_overrides[get_db] = database
    try:
        with TestClient(app) as client:
            started = client.post("/api/tasks/start", json={"request": "Investigate using summarize_numbers, calculate, convert_units for the demo", "tool_scope": "NUMERIC"})
            assert started.status_code == 202
            events = client.get(f"/api/tasks/{started.json()['task_id']}/events").text
            completed = [json.loads(line[6:]) for line in events.splitlines() if line.startswith("data: ")]
            final = next(item["payload"]["result"] for item in completed if item["type"] == "task_completed")
            timing = final["runtime_metrics"]["timing"]
            selection = final["workflow_selection"]
            selected_events = [item for item in completed if item["type"] == "workflow_selected"]
            assert len(selected_events) == 1 and selected_events[0]["payload"] == selection
            assert selection["strategy"] == "ADAPTIVE" and not selection["planning_candidate"]
            assert completed.index(selected_events[0]) < next(i for i, item in enumerate(completed) if item["type"] == "tool_started")
            assert timing["llm_call_count"] == 4 and len(timing["tools"]) == 3
            assert [call["purpose"] for call in timing["llm_calls"]] == [f"tool_decision_{i}" for i in range(1, 5)]
            assert all(call["prompt_token_count"] == 120 for call in timing["llm_calls"])
            assert set(timing["stages_seconds"]) == {"input_checks", "execution", "grounding_and_governance", "persistence_and_audit"}
            assert sum(timing["stages_seconds"].values()) <= timing["task_handler_seconds"] + .00001
            restored = client.get(f"/api/tasks/{final['id']}")
            assert restored.status_code == 200
            assert restored.json()["runtime_metrics"]["timing"] == timing
            assert restored.json()["workflow_selection"] == selection
            assert final["governance"]["policy"] == "internal_assistant"
    finally:
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()
        get_settings.cache_clear()
