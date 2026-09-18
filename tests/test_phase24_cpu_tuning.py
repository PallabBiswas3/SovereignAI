import asyncio
import json

import httpx

from app.llm.ollama_provider import OllamaProvider
from app.resources.cpu_tuning import CpuTuningProfile, CpuTuningStore
from app.resources.lifecycle import ModelLifecycleManager
from app.resources.scheduler import ResourceScheduler


def test_cpu_tuning_store_round_trip(tmp_path) -> None:
    store = CpuTuningStore(tmp_path / "cpu_tuning.json")
    store.save(CpuTuningProfile(
        model="qwen3:4b-instruct",
        num_thread=5,
        num_batch=64,
        num_ctx=4096,
        median_tokens_per_second=12.2,
        mean_tokens_per_second=12.0,
        measured_runs=3,
        warmup_runs=1,
    ))
    profile = store.get("QWEN3:4B-INSTRUCT")
    assert profile is not None
    assert profile.num_thread == 5
    assert profile.num_batch == 64
    assert store.options_for("qwen3:4b-instruct") == {"num_thread": 5, "num_batch": 64}


def test_ollama_provider_applies_cpu_tuning_without_overriding_runtime_budget(tmp_path) -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3:4b-instruct"}]})
        payload = json.loads(request.content)
        requests.append(payload)
        body = (
            '{"model":"qwen3:4b-instruct","response":"ok","done":false}\n'
            '{"model":"qwen3:4b-instruct","response":"","done":true,'
            '"total_duration":1000000000,"load_duration":10000000,'
            '"eval_duration":500000000,"eval_count":5,"prompt_eval_count":3}\n'
        )
        return httpx.Response(200, content=body.encode("utf-8"))

    async def run() -> dict[str, object]:
        tuning = CpuTuningStore(tmp_path / "cpu_tuning.json")
        tuning.save(CpuTuningProfile(
            model="qwen3:4b-instruct",
            num_thread=5,
            num_batch=64,
            num_ctx=4096,
            median_tokens_per_second=12.0,
        ))
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OllamaProvider(
            "http://127.0.0.1:11434",
            allow_fallback=False,
            client=client,
            scheduler=ResourceScheduler(ram_admission_enabled=False),
            lifecycle=ModelLifecycleManager(),
            cpu_tuning=tuning,
            execution_mode="STANDARD",
        )
        provider.adaptive_residency_enabled = False
        provider.adaptive_context_enabled = False
        chunks = [chunk async for chunk in provider.stream("test", "qwen3:4b-instruct")]
        await client.aclose()
        return chunks[-1].runtime_stats

    stats = asyncio.run(run())
    options = requests[0]["options"]
    assert options["num_thread"] == 5
    assert options["num_batch"] == 64
    assert options["num_ctx"] == 8192
    assert options["num_predict"] == 1024
    assert stats["cpu_tuning"]["num_thread"] == 5
    assert stats["cpu_tuning"]["num_batch"] == 64
