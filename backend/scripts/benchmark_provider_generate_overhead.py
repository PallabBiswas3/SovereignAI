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
import psutil

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
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


def _memory_snapshot() -> dict[str, float]:
    memory = psutil.virtual_memory()
    return {
        "available_mb": round(memory.available / 1024 / 1024, 2),
        "used_mb": round(memory.used / 1024 / 1024, 2),
        "total_mb": round(memory.total / 1024 / 1024, 2),
        "percent": round(float(memory.percent), 2),
    }


def _ram_threshold(headroom_mb: float) -> tuple[float, float]:
    settings = get_settings()
    memory = psutil.virtual_memory()
    total_mb = memory.total / 1024 / 1024
    reserve_mb = max(float(settings.model_ram_reserve_mb), total_mb * float(settings.model_ram_reserve_fraction))
    return reserve_mb, reserve_mb + max(0.0, headroom_mb)


async def _wait_for_ram(minimum_available_mb: float, timeout_seconds: float) -> dict[str, Any]:
    started = perf_counter()
    deadline = asyncio.get_running_loop().time() + max(0.0, timeout_seconds)
    snapshot = _memory_snapshot()
    while snapshot["available_mb"] < minimum_available_mb and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.5)
        snapshot = _memory_snapshot()
    return {
        "ready": snapshot["available_mb"] >= minimum_available_mb,
        "wait_seconds": round(perf_counter() - started, 6),
        "minimum_available_mb": round(minimum_available_mb, 2),
        "memory": snapshot,
    }


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


async def _resident_names(client: httpx.AsyncClient, endpoint: str) -> set[str]:
    response = await client.get(f"{endpoint.rstrip('/')}/api/ps", timeout=5.0)
    response.raise_for_status()
    return {
        str(item.get("name") or item.get("model") or "").strip()
        for item in response.json().get("models", [])
        if isinstance(item, dict)
    }


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
            "temperature": 0.2,
            "num_thread": 5,
            "num_batch": 128,
        },
    }
    started = perf_counter()
    response = await client.post(f"{endpoint.rstrip('/')}/api/generate", json=payload, timeout=timeout)
    wall = perf_counter() - started
    response.raise_for_status()
    data = response.json()
    names = await _resident_names(client, endpoint)
    resident = model in names or (":" not in model and f"{model}:latest" in names)
    if not resident:
        raise RuntimeError(f"{model} did not become resident after exact-runner warm-up")
    return {
        "wall_seconds": round(wall, 6),
        "load_duration_seconds": round(float(data.get("load_duration", 0) or 0) / 1_000_000_000, 6),
        "memory_after": _memory_snapshot(),
    }


async def _reset_and_warm(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    timeout: float,
    settle_seconds: float,
) -> dict[str, Any]:
    before = _memory_snapshot()
    response = await client.post(
        f"{endpoint.rstrip('/')}/api/generate",
        json={"model": model, "stream": False, "keep_alive": 0},
        timeout=timeout,
    )
    response.raise_for_status()

    deadline = asyncio.get_running_loop().time() + 15.0
    while asyncio.get_running_loop().time() < deadline:
        names = await _resident_names(client, endpoint)
        if model not in names and (":" in model or f"{model}:latest" not in names):
            break
        await asyncio.sleep(0.25)
    names = await _resident_names(client, endpoint)
    if model in names or (":" not in model and f"{model}:latest" in names):
        raise RuntimeError(f"{model} remained resident after unload request")

    if settle_seconds > 0:
        await asyncio.sleep(settle_seconds)
    after_unload = _memory_snapshot()
    warm = await _ensure_warm(client, endpoint, model, timeout)
    return {
        "memory_before_reset": before,
        "memory_after_unload": after_unload,
        "exact_runner_warmup": warm,
        "memory_after_warm": _memory_snapshot(),
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


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
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


def _build_report(
    *,
    args: argparse.Namespace,
    reserve_mb: float,
    minimum_available_mb: float,
    rounds: list[dict[str, Any]],
    fresh_runs: list[dict[str, Any]],
    shared_runs: list[dict[str, Any]],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "benchmark_design": "actual-ollama-provider-generate-wrapper-overhead-v2-ram-stable",
        "status": status,
        "error": error,
        "model": args.model,
        "requested_repeats": max(4, args.repeats),
        "completed_rounds": len(rounds),
        "ram_reserve_mb": round(reserve_mb, 2),
        "ram_headroom_mb": args.ram_headroom_mb,
        "minimum_benchmark_available_mb": round(minimum_available_mb, 2),
        "rounds": rounds,
        "summary": {
            "fresh_client_provider": _summarize(fresh_runs),
            "shared_client_provider": _summarize(shared_runs),
        },
        "interpretation_note": (
            "Every comparison round resets the target model, performs an exact-runner warm-up outside measured "
            "calls, and waits for available RAM to exceed the unchanged production reserve plus benchmark-only "
            "headroom. The measured calls use the real OllamaProvider.generate() path."
        ),
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    _require_local(args.ollama_url)
    repeats = max(4, args.repeats)
    timeout = httpx.Timeout(args.timeout)
    reserve_mb, minimum_available_mb = _ram_threshold(args.ram_headroom_mb)

    fresh_runs: list[dict[str, Any]] = []
    shared_runs: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as control_client, httpx.AsyncClient(
        timeout=timeout, follow_redirects=False
    ) as shared_client:
        for index in range(repeats):
            round_record: dict[str, Any] = {
                "round": index + 1,
                "reset": await _reset_and_warm(
                    control_client,
                    args.ollama_url,
                    args.model,
                    args.timeout,
                    max(0.0, args.settle_seconds),
                ),
            }
            gate = await _wait_for_ram(minimum_available_mb, args.ram_wait_seconds)
            round_record["pre_round_ram_gate"] = gate
            if not gate["ready"]:
                rounds.append(round_record)
                report = _build_report(
                    args=args,
                    reserve_mb=reserve_mb,
                    minimum_available_mb=minimum_available_mb,
                    rounds=rounds,
                    fresh_runs=fresh_runs,
                    shared_runs=shared_runs,
                    status="blocked_by_ram",
                    error=(
                        f"Available RAM did not reach {minimum_available_mb:.0f} MB after reset/warm. "
                        "Close memory-heavy applications and rerun; the production reserve was not changed."
                    ),
                )
                return report

            order = ["fresh", "shared"] if index % 2 == 0 else ["shared", "fresh"]
            round_record["order"] = order
            for target in order:
                target_gate = await _wait_for_ram(minimum_available_mb, args.ram_wait_seconds)
                round_record[f"ram_gate_before_{target}"] = target_gate
                if not target_gate["ready"]:
                    rounds.append(round_record)
                    return _build_report(
                        args=args,
                        reserve_mb=reserve_mb,
                        minimum_available_mb=minimum_available_mb,
                        rounds=rounds,
                        fresh_runs=fresh_runs,
                        shared_runs=shared_runs,
                        status="blocked_by_ram",
                        error=(
                            f"Available RAM fell below benchmark headroom before {target}. "
                            "Close memory-heavy applications and rerun; the production reserve was not changed."
                        ),
                    )
                run = await _run_provider_once(
                    endpoint=args.ollama_url,
                    model=args.model,
                    prompt=args.prompt,
                    shared_client=shared_client if target == "shared" else None,
                )
                run["client_mode"] = target
                round_record[target] = run
                (shared_runs if target == "shared" else fresh_runs).append(run)

            round_record["memory_after_round"] = _memory_snapshot()
            rounds.append(round_record)

            progress = _build_report(
                args=args,
                reserve_mb=reserve_mb,
                minimum_available_mb=minimum_available_mb,
                rounds=rounds,
                fresh_runs=fresh_runs,
                shared_runs=shared_runs,
                status="running" if len(rounds) < repeats else "complete",
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(progress, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return _build_report(
        args=args,
        reserve_mb=reserve_mb,
        minimum_available_mb=minimum_available_mb,
        rounds=rounds,
        fresh_runs=fresh_runs,
        shared_runs=shared_runs,
        status="complete",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure actual OllamaProvider.generate wrapper overhead.")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--ram-headroom-mb", type=float, default=96.0)
    parser.add_argument("--ram-wait-seconds", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=Path("workspace/provider_generate_overhead.json"))
    args = parser.parse_args()

    report = asyncio.run(_run(args))
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if report.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
