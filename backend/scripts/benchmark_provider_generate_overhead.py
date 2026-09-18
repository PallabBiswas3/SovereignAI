from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

import httpx

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.llm.ollama_provider import OllamaProvider

SYSTEM_PROMPT = "You are SovereignAI, a local enterprise assistant. Be concise and never invent sources."
DEFAULT_PROMPT = (
    "Answer with exactly four short bullets, one sentence each: why does an outer-race "
    "bearing fault create periodic vibration impulses?"
)


def _require_local(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"Benchmark endpoint must be loopback-local: {url}")


def _median(values: list[float]) -> float | None:
    return round(median(values), 6) if values else None


class _TimedLifecycle:
    """Proxy lifecycle manager that records exact call cost without changing behavior."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.begin_seconds = 0.0
        self.mark_busy_seconds = 0.0
        self.complete_seconds = 0.0
        self.fail_seconds = 0.0
        self.mark_busy_calls = 0

    def begin(self, *args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        try:
            return self.inner.begin(*args, **kwargs)
        finally:
            self.begin_seconds += perf_counter() - started

    def mark_busy(self, *args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        try:
            return self.inner.mark_busy(*args, **kwargs)
        finally:
            self.mark_busy_seconds += perf_counter() - started
            self.mark_busy_calls += 1

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        try:
            return self.inner.complete(*args, **kwargs)
        finally:
            self.complete_seconds += perf_counter() - started

    def fail(self, *args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        try:
            return self.inner.fail(*args, **kwargs)
        finally:
            self.fail_seconds += perf_counter() - started

    def get(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.get(*args, **kwargs)

    def all(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.all(*args, **kwargs)


async def _ensure_warm(client: httpx.AsyncClient, endpoint: str, model: str, timeout: float) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": "/no_think\nReply with exactly READY.",
        "stream": False,
        "think": False,
        "keep_alive": "15m",
        "options": {
            "num_ctx": 2048,
            "num_predict": 1,
            "temperature": 0.0,
            "num_thread": 5,
            "num_batch": 128,
        },
    }
    started = perf_counter()
    response = await client.post(f"{endpoint.rstrip('/')}/api/generate", json=payload, timeout=timeout)
    wall = perf_counter() - started
    response.raise_for_status()
    data = response.json()
    return {
        "wall_seconds": round(wall, 6),
        "load_duration_seconds": round(float(data.get("load_duration", 0) or 0) / 1_000_000_000, 6),
    }


async def _run_provider_once(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    shared_client: httpx.AsyncClient | None,
) -> dict[str, Any]:
    construct_started = perf_counter()
    provider = OllamaProvider(
        endpoint,
        allow_fallback=False,
        client=shared_client,
        role="GENERAL",
        memory_requirement="medium",
        execution_mode="FAST",
    )
    provider_construct_seconds = perf_counter() - construct_started

    residency_seconds = 0.0
    original_prepare = provider._prepare_residency

    async def timed_prepare(target_model: str):
        nonlocal residency_seconds
        started = perf_counter()
        try:
            return await original_prepare(target_model)
        finally:
            residency_seconds += perf_counter() - started

    provider._prepare_residency = timed_prepare  # type: ignore[method-assign]
    timed_lifecycle = _TimedLifecycle(provider.lifecycle)
    provider.lifecycle = timed_lifecycle  # type: ignore[assignment]

    started = perf_counter()
    result = await provider.generate(prompt, model, SYSTEM_PROMPT)
    provider_wall_seconds = perf_counter() - started

    stats = result.runtime_stats if isinstance(result.runtime_stats, dict) else {}
    server_total = float(stats.get("total_duration_seconds", 0) or 0)
    queue_wait = float(stats.get("queue_wait_seconds", 0) or 0)
    wrapper_overhead = provider_wall_seconds - server_total if server_total else None
    lifecycle_total = (
        timed_lifecycle.begin_seconds
        + timed_lifecycle.mark_busy_seconds
        + timed_lifecycle.complete_seconds
        + timed_lifecycle.fail_seconds
    )
    known_timed = residency_seconds + queue_wait + lifecycle_total
    residual = wrapper_overhead - known_timed if wrapper_overhead is not None else None

    return {
        "provider_construct_seconds": round(provider_construct_seconds, 6),
        "provider_wall_seconds": round(provider_wall_seconds, 6),
        "ollama_total_duration_seconds": round(server_total, 6) if server_total else None,
        "provider_wrapper_overhead_seconds": round(wrapper_overhead, 6) if wrapper_overhead is not None else None,
        "residency_prepare_seconds": round(residency_seconds, 6),
        "queue_wait_seconds": round(queue_wait, 6),
        "lifecycle_begin_seconds": round(timed_lifecycle.begin_seconds, 6),
        "lifecycle_mark_busy_seconds": round(timed_lifecycle.mark_busy_seconds, 6),
        "lifecycle_complete_seconds": round(timed_lifecycle.complete_seconds, 6),
        "lifecycle_mark_busy_calls": timed_lifecycle.mark_busy_calls,
        "known_timed_wrapper_seconds": round(known_timed, 6),
        "residual_wrapper_seconds": round(residual, 6) if residual is not None else None,
        "time_to_first_token_seconds": stats.get("time_to_first_token_seconds"),
        "tokens_per_second": stats.get("tokens_per_second"),
        "token_count": stats.get("token_count"),
        "load_duration_seconds": stats.get("load_duration_seconds"),
        "warm_status": stats.get("warm_status"),
        "fallback": result.fallback,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    _require_local(args.ollama_url)
    repeats = max(4, args.repeats)
    timeout = httpx.Timeout(args.timeout)

    fresh_runs: list[dict[str, Any]] = []
    shared_runs: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as warm_client:
        warmup = await _ensure_warm(warm_client, args.ollama_url, args.model, args.timeout)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as shared_client:
        for index in range(repeats):
            order = ["fresh", "shared"] if index % 2 == 0 else ["shared", "fresh"]
            for target in order:
                run = await _run_provider_once(
                    endpoint=args.ollama_url,
                    model=args.model,
                    prompt=args.prompt,
                    shared_client=shared_client if target == "shared" else None,
                )
                run["client_mode"] = target
                (shared_runs if target == "shared" else fresh_runs).append(run)

    def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        keys = (
            "provider_construct_seconds",
            "provider_wall_seconds",
            "ollama_total_duration_seconds",
            "provider_wrapper_overhead_seconds",
            "residency_prepare_seconds",
            "queue_wait_seconds",
            "lifecycle_begin_seconds",
            "lifecycle_mark_busy_seconds",
            "lifecycle_complete_seconds",
            "known_timed_wrapper_seconds",
            "residual_wrapper_seconds",
            "time_to_first_token_seconds",
            "tokens_per_second",
            "token_count",
            "load_duration_seconds",
        )
        for key in keys:
            values = [float(run[key]) for run in runs if isinstance(run.get(key), (int, float))]
            out[f"median_{key}"] = _median(values)
        return out

    return {
        "benchmark_design": "actual-ollama-provider-generate-wrapper-overhead-v1",
        "model": args.model,
        "repeats": repeats,
        "warmup": warmup,
        "fresh_client_provider_runs": fresh_runs,
        "shared_client_provider_runs": shared_runs,
        "summary": {
            "fresh_client_provider": summarize(fresh_runs),
            "shared_client_provider": summarize(shared_runs),
        },
        "interpretation_note": (
            "This benchmark calls the real OllamaProvider.generate() path used by /api/chat. It measures provider "
            "wall time against Ollama's own total_duration and separately times adaptive residency, scheduler queue "
            "wait, and lifecycle bookkeeping. The residual is intentionally diagnostic: it contains remaining "
            "client creation/stream parsing/chunk-yield/stats/cleanup work plus timing-definition differences."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure actual OllamaProvider.generate wrapper overhead.")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=Path("workspace/provider_generate_overhead.json"))
    args = parser.parse_args()

    report = asyncio.run(_run(args))
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
