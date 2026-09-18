import asyncio
import json

import httpx

from app.llm.ollama_provider import OllamaProvider
from app.resources.context_planner import AdaptiveContextPlanner
from app.resources.lifecycle import ModelLifecycleManager
from app.resources.model_optimization import ModelOptimizationProfile, ModelOptimizationStore
from app.resources.scheduler import ResourceScheduler


def test_context_planner_uses_smallest_safe_bucket() -> None:
    planner = AdaptiveContextPlanner(
        buckets=(2048, 4096, 8192),
        minimum_context=2048,
        bytes_per_token=3.0,
        prompt_margin_tokens=128,
    )
    short = planner.plan(
        prompt="Explain pump cavitation briefly.",
        system=None,
        execution_mode="FAST",
        profile_max_num_ctx=4096,
        output_reserve_tokens=384,
    )
    assert short.selected_num_ctx == 2048
    assert short.truncated_risk is False

    medium = planner.plan(
        prompt="x" * 6000,
        system=None,
        execution_mode="FAST",
        profile_max_num_ctx=4096,
        output_reserve_tokens=384,
    )
    assert medium.selected_num_ctx == 4096
    assert medium.truncated_risk is False


def test_context_planner_flags_profile_ceiling_risk() -> None:
    planner = AdaptiveContextPlanner(buckets=(2048, 4096), minimum_context=2048)
    plan = planner.plan(
        prompt="x" * 20000,
        system=None,
        execution_mode="FAST",
        profile_max_num_ctx=4096,
        output_reserve_tokens=384,
    )
    assert plan.selected_num_ctx == 4096
    assert plan.truncated_risk is True


def test_model_optimization_store_round_trip(tmp_path) -> None:
    store = ModelOptimizationStore(tmp_path / "model_optimization.json")
    store.save(ModelOptimizationProfile(
        base_model="qwen3:4b-instruct",
        selected_model="qwen3:4b-instruct-q3km",
        quantization="Q3_K_M",
        quality_score=1.0,
        tokens_per_second=13.0,
        resident_size_mb=2400,
        context_length=2048,
    ))
    profile = store.get("QWEN3:4B-INSTRUCT")
    assert profile is not None
    assert profile.selected_model == "qwen3:4b-instruct-q3km"
    assert store.selected_model("qwen3:4b-instruct") == "qwen3:4b-instruct-q3km"


def test_provider_applies_adaptive_context_and_effective_model(tmp_path) -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3:4b-instruct-q3km"}]})
        payload = json.loads(request.content)
        requests.append(payload)
        body = (
            '{"model":"qwen3:4b-instruct-q3km","response":"ok","done":false}\n'
            '{"model":"qwen3:4b-instruct-q3km","response":"","done":true,'
            '"total_duration":1000000000,"load_duration":10000000,'
            '"eval_duration":500000000,"eval_count":5,"prompt_eval_count":3}\n'
        )
        return httpx.Response(200, content=body.encode("utf-8"))

    async def run():
        optimization = ModelOptimizationStore(tmp_path / "model_optimization.json")
        optimization.save(ModelOptimizationProfile(
            base_model="qwen3:4b-instruct",
            selected_model="qwen3:4b-instruct-q3km",
            quantization="Q3_K_M",
            quality_score=1.0,
            tokens_per_second=13.0,
            resident_size_mb=2400,
            context_length=2048,
        ))
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OllamaProvider(
            "http://127.0.0.1:11434",
            allow_fallback=False,
            client=client,
            scheduler=ResourceScheduler(ram_admission_enabled=False),
            lifecycle=ModelLifecycleManager(),
            model_optimization=optimization,
            execution_mode="FAST",
        )
        provider.adaptive_residency_enabled = False
        chunks = [chunk async for chunk in provider.stream("short prompt", "qwen3:4b-instruct")]
        await client.aclose()
        return chunks

    chunks = asyncio.run(run())
    payload = requests[0]
    assert payload["model"] == "qwen3:4b-instruct-q3km"
    assert payload["options"]["num_ctx"] == 2048
    assert chunks[-1].model == "qwen3:4b-instruct"
    stats = chunks[-1].runtime_stats
    assert stats["requested_model"] == "qwen3:4b-instruct"
    assert stats["effective_model"] == "qwen3:4b-instruct-q3km"
    assert stats["context_plan"]["selected_num_ctx"] == 2048
    assert stats["model_optimization"]["quantization"] == "Q3_K_M"
