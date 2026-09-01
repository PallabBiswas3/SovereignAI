import asyncio
import json

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.llm.base import ModelGenerationCancelled
from app.llm.ollama_provider import OllamaProvider
from app.orchestration.execution_mode import ExecutionMode, ExecutionModeSelector
from app.rag.embeddings import CachedEmbeddingProvider, LocalHashEmbeddingProvider
from app.resources.cache import CacheKeyBuilder, CacheNamespace, SQLiteCache
from app.resources.lifecycle import ModelLifecycleManager, ModelLifecycleState
from app.resources.scheduler import ModelJob, ResourceScheduler
from app.router.schemas import TaskProfile


def _profile(task_type: str = "general") -> TaskProfile:
    return TaskProfile(
        task_type=task_type,
        coding_requirement=0.0,
        reasoning_requirement=0.4,
        vision_requirement=0.0,
        document_requirement=0.0,
        summarization_requirement=0.0,
        latency_priority=0.5,
        context_length_required=1024,
    )


def test_execution_mode_selection_is_explicit_and_auditable() -> None:
    selector = ExecutionModeSelector()
    fast = selector.select(ExecutionMode.automatic, "What is the pressure limit?", _profile())
    deep = selector.select(ExecutionMode.automatic, "Investigate this failure", _profile(), 2)
    explicit = selector.select(ExecutionMode.standard, "What is the pressure limit?", _profile())
    assert fast.selected == ExecutionMode.fast
    assert deep.selected == ExecutionMode.deep
    assert explicit.selected == ExecutionMode.standard
    assert explicit.reason


def test_ollama_provider_streams_ndjson_and_records_metrics() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        payload = json.loads(request.content)
        requests.append(payload)
        body = (
            '{"model":"local:test","response":"Pump ","done":false}\n'
            '{"model":"local:test","response":"ready.","done":false}\n'
            '{"model":"local:test","response":"","done":true,"total_duration":2000000000,'
            '"load_duration":100000000,"eval_duration":1000000000,"eval_count":2,'
            '"prompt_eval_count":4}\n'
        )
        return httpx.Response(200, content=body.encode("utf-8"))

    async def run() -> tuple[list[str], dict[str, object]]:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OllamaProvider(
            "http://127.0.0.1:11434",
            allow_fallback=False,
            client=client,
            scheduler=ResourceScheduler(),
            lifecycle=ModelLifecycleManager(),
        )
        chunks = [chunk async for chunk in provider.stream("Inspect pump", "local:test")]
        await client.aclose()
        return [chunk.text for chunk in chunks if chunk.text], chunks[-1].runtime_stats

    pieces, stats = asyncio.run(run())
    assert pieces == ["Pump ", "ready."]
    assert requests[0]["stream"] is True
    assert requests[0]["keep_alive"]
    assert stats["token_count"] == 2
    assert stats["tokens_per_second"] == 2.0
    assert stats["warm_status"] == "cold"


def test_generation_honors_preexisting_cancellation() -> None:
    async def run() -> None:
        event = asyncio.Event()
        event.set()
        provider = OllamaProvider(
            "http://127.0.0.1:11434",
            allow_fallback=False,
            scheduler=ResourceScheduler(),
            lifecycle=ModelLifecycleManager(),
        )
        try:
            async for _ in provider.stream("test", "local:test", cancellation_event=event):
                pass
        except ModelGenerationCancelled:
            return
        raise AssertionError("Expected generation cancellation")

    asyncio.run(run())


def test_scheduler_serializes_gpu_model_jobs() -> None:
    async def run() -> tuple[int, int]:
        scheduler = ResourceScheduler(max_gpu_model_jobs=1)
        release = asyncio.Event()
        entered = asyncio.Event()
        maximum_active = 0

        async def first() -> None:
            nonlocal maximum_active
            async with scheduler.acquire_model(ModelJob(model="first")):
                maximum_active = max(maximum_active, scheduler.snapshot().active_gpu_jobs)
                entered.set()
                await release.wait()

        async def second() -> None:
            nonlocal maximum_active
            await entered.wait()
            async with scheduler.acquire_model(ModelJob(model="second", priority=100)):
                maximum_active = max(maximum_active, scheduler.snapshot().active_gpu_jobs)

        first_task = asyncio.create_task(first())
        second_task = asyncio.create_task(second())
        await entered.wait()
        await asyncio.sleep(0)
        queued = scheduler.queue_depth
        release.set()
        await asyncio.gather(first_task, second_task)
        return maximum_active, queued

    maximum_active, queued = asyncio.run(run())
    assert maximum_active == 1
    assert queued == 1


def test_model_lifecycle_reports_warmth_and_performance() -> None:
    lifecycle = ModelLifecycleManager()
    assert lifecycle.observe("model", installed=False, loaded=False).state == ModelLifecycleState.not_installed
    lifecycle.begin("model", was_loaded=False, queue_depth=2)
    assert lifecycle.get("model").state == ModelLifecycleState.loading
    lifecycle.mark_busy("model")
    lifecycle.complete("model", {
        "time_to_first_token_seconds": 0.25,
        "tokens_per_second": 12.5,
        "total_duration_seconds": 1.0,
        "load_duration_seconds": 0.1,
        "token_count": 10,
    })
    snapshot = lifecycle.get("model")
    assert snapshot.state == ModelLifecycleState.idle
    assert snapshot.tokens_per_second == 12.5
    assert snapshot.token_count == 10


def test_versioned_sqlite_cache_and_cached_embeddings(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'cache.db'}")
    Base.metadata.create_all(engine)
    cache = SQLiteCache(sessionmaker(bind=engine))
    key_v1 = CacheKeyBuilder.embedding("pump", "hash:v1")
    key_v2 = CacheKeyBuilder.embedding("pump", "hash:v2")
    cache.set(CacheNamespace.embedding.value, key_v1, [1.0, 0.0])
    assert cache.get(CacheNamespace.embedding.value, key_v1) == [1.0, 0.0]
    assert cache.get(CacheNamespace.embedding.value, key_v2) is None

    cached_provider = CachedEmbeddingProvider(LocalHashEmbeddingProvider(16), cache)
    first = cached_provider.embed_query("bearing temperature")
    second = cached_provider.embed_query("bearing temperature")
    assert first == second
    assert cache.stats()["hits"] >= 2
