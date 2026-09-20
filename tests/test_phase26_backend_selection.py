import asyncio
import json

import httpx

from app.core.config import Settings
from app.llm.llama_cpp_provider import LlamaCppProvider
from app.llm.provider_factory import create_local_model_provider
from app.resources.backend_selection import BackendSelectionProfile, BackendSelectionStore
from app.resources.lifecycle import ModelLifecycleManager
from app.resources.scheduler import ResourceScheduler


def test_backend_selection_store_round_trip(tmp_path) -> None:
    store = BackendSelectionStore(tmp_path / "backend_selection.json")
    store.save(BackendSelectionProfile(
        model="qwen3:4b-instruct",
        backend="llama_cpp",
        endpoint="http://127.0.0.1:8080",
        backend_model="qwen3:4b-instruct",
        quality_score=1.0,
        median_tokens_per_second=12.5,
        median_wall_seconds=10.0,
        speedup_vs_ollama=1.12,
    ))
    profile = store.get("QWEN3:4B-INSTRUCT")
    assert profile is not None
    assert profile.backend == "llama_cpp"
    assert profile.endpoint == "http://127.0.0.1:8080"
    assert profile.backend_model == "qwen3:4b-instruct"


def test_provider_factory_defaults_to_ollama_and_honors_validated_llama_cpp(tmp_path) -> None:
    settings = Settings(
        model_backend_selection_path=tmp_path / "backend_selection.json",
        model_backend_selection_enabled=True,
        llama_cpp_url="http://127.0.0.1:8080",
    )
    default_provider = create_local_model_provider(
        settings,
        model="qwen3:4b-instruct",
        ollama_endpoint="http://127.0.0.1:11434",
    )
    assert default_provider.__class__.__name__ == "OllamaProvider"

    BackendSelectionStore(settings.model_backend_selection_path).save(BackendSelectionProfile(
        model="qwen3:4b-instruct",
        backend="llama_cpp",
        endpoint="http://127.0.0.1:8080",
        backend_model="qwen3:4b-instruct",
        quality_score=1.0,
        median_tokens_per_second=12.5,
        speedup_vs_ollama=1.10,
    ))
    selected_provider = create_local_model_provider(
        settings,
        model="qwen3:4b-instruct",
        ollama_endpoint="http://127.0.0.1:11434",
    )
    assert isinstance(selected_provider, LlamaCppProvider)
    assert selected_provider.endpoint == "http://127.0.0.1:8080"


def test_llama_cpp_provider_streams_sse_and_records_timings() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            payload = json.loads(request.content)
            requests.append(payload)
            body = (
                'data: {"model":"qwen3:4b-instruct","choices":[{"delta":{"content":"Pump "},"finish_reason":null}]}\n\n'
                'data: {"model":"qwen3:4b-instruct","choices":[{"delta":{"content":"ready."},"finish_reason":null}]}\n\n'
                'data: {"model":"qwen3:4b-instruct","choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":4,"completion_tokens":2},'
                '"timings":{"prompt_n":4,"prompt_per_second":100.0,"predicted_n":2,"predicted_per_second":10.5}}\n\n'
                'data: [DONE]\n\n'
            )
            return httpx.Response(200, content=body.encode("utf-8"), headers={"content-type": "text/event-stream"})
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen3:4b-instruct"}]})
        raise AssertionError(f"Unexpected path: {request.url.path}")

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = LlamaCppProvider(
            "http://127.0.0.1:8080",
            allow_fallback=False,
            client=client,
            scheduler=ResourceScheduler(ram_admission_enabled=False),
            lifecycle=ModelLifecycleManager(),
            execution_mode="FAST",
            server_context_size=4096,
        )
        chunks = [chunk async for chunk in provider.stream("Inspect pump", "qwen3:4b-instruct")]
        await client.aclose()
        return chunks

    chunks = asyncio.run(run())
    assert [chunk.text for chunk in chunks if chunk.text] == ["Pump ", "ready."]
    assert requests[0]["stream"] is True
    assert requests[0]["max_tokens"] == 384
    assert requests[0]["chat_template_kwargs"]["enable_thinking"] is False
    stats = chunks[-1].runtime_stats
    assert stats["backend"] == "llama_cpp"
    assert stats["tokens_per_second"] == 10.5
    assert stats["prompt_tokens_per_second"] == 100.0
    assert stats["token_count"] == 2
    assert stats["done_reason"] == "stop"
