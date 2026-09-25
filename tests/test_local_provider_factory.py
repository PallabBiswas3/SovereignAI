from __future__ import annotations

import asyncio
import json

import httpx

from app.core.config import Settings
from app.llm.factory import configured_local_provider
from app.llm.ollama_provider import OllamaProvider
from app.llm.vllm_provider import VLLMProvider
from app.resources.lifecycle import ModelLifecycleManager, ModelLifecycleState
from app.resources.scheduler import ResourceScheduler
from app.router.schemas import ModelDefinition


def _model() -> ModelDefinition:
    return ModelDefinition(
        id="general",
        role="GENERAL",
        display_name="General",
        provider="ollama",
        model_tag="qwen3:4b-instruct",
        endpoint="http://127.0.0.1:11434",
        capabilities={"reasoning": 0.9},
        context_length=8192,
        memory_requirement="medium",
    )


def test_factory_preserves_routed_ollama_model() -> None:
    runtime = configured_local_provider(
        Settings(llm_provider="ollama"),
        model_definition=_model(),
        execution_mode="DEEP",
        priority=80,
    )

    assert isinstance(runtime.provider, OllamaProvider)
    assert runtime.model == "qwen3:4b-instruct"
    assert runtime.endpoint == "http://127.0.0.1:11434"
    assert runtime.provider.execution_mode == "DEEP"
    assert runtime.provider.priority == 80


def test_factory_selects_vllm_and_global_served_model() -> None:
    runtime = configured_local_provider(
        Settings(
            llm_provider="vllm",
            vllm_url="http://127.0.0.1:8001",
            vllm_model="Qwen/Qwen3-0.6B",
            vllm_enable_thinking=False,
            vllm_max_tokens=100,
        ),
        model_definition=_model(),
        execution_mode="STANDARD",
        priority=60,
    )

    assert isinstance(runtime.provider, VLLMProvider)
    assert runtime.model == "Qwen/Qwen3-0.6B"
    assert runtime.endpoint == "http://127.0.0.1:8001/v1"
    assert runtime.provider.priority == 60
    assert runtime.provider._base_payload("hello", runtime.model, "system") == {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ],
        "temperature": 0.2,
        "max_tokens": 100,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_factory_rejects_unknown_provider() -> None:
    try:
        configured_local_provider(Settings(llm_provider="remote-cloud"))
    except ValueError as exc:
        assert "Unsupported local model provider" in str(exc)
    else:
        raise AssertionError("Unknown provider configuration must fail closed")


def test_vllm_stream_records_scheduler_and_lifecycle_metrics() -> None:
    model = "Qwen/Qwen3-0.6B"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": model}]})
        assert request.url.path == "/v1/chat/completions"
        body = "\n".join([
            f"data: {json.dumps({'choices': [{'delta': {'content': 'supported'}}]})}",
            f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': 8, 'completion_tokens': 1}})}",
            "data: [DONE]",
            "",
        ])
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    lifecycle = ModelLifecycleManager()
    scheduler = ResourceScheduler(max_gpu_model_jobs=1)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = VLLMProvider(
        "http://127.0.0.1:8001",
        client=client,
        scheduler=scheduler,
        lifecycle=lifecycle,
    )

    try:
        result = asyncio.run(provider.generate("question", model))
    finally:
        asyncio.run(client.aclose())

    snapshot = lifecycle.get(model)
    assert result.text == "supported"
    assert result.runtime_stats["queue_wait_seconds"] >= 0
    assert result.runtime_stats["prompt_token_count"] == 8
    assert snapshot.state == ModelLifecycleState.idle
    assert snapshot.attempt_count == 1
    assert scheduler.snapshot().active_gpu_jobs == 0
