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


def _require_local(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"Benchmark endpoint must be loopback-local: {url}")


def _median(values: list[float]) -> float | None:
    return round(median(values), 6) if values else None


async def _measure_preflight(
    *,
    endpoint: str,
    model: str,
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
    construct_seconds = perf_counter() - construct_started

    residency_started = perf_counter()
    decision = await provider._prepare_residency(model)  # benchmark-only introspection
    residency_seconds = perf_counter() - residency_started

    scheduler_started = perf_counter()
    async with provider.scheduler.acquire_model(provider._job(model, decision)) as permit:  # benchmark-only introspection
        scheduler_admission_seconds = perf_counter() - scheduler_started

    return {
        "provider_construct_seconds": round(construct_seconds, 6),
        "residency_prepare_seconds": round(residency_seconds, 6),
        "scheduler_admission_seconds": round(scheduler_admission_seconds, 6),
        "preflight_total_seconds": round(
            construct_seconds + residency_seconds + scheduler_admission_seconds, 6
        ),
        "target_was_resident": bool(decision and decision.target_was_resident),
        "ram_available_mb": decision.available_before_mb if decision else None,
        "ram_reserve_mb": decision.reserve_mb if decision else None,
        "scheduler_queue_wait_seconds": permit.queue_wait_seconds,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    _require_local(args.ollama_url)
    repeats = max(3, int(args.repeats))

    # Verify the server and target model residency first. This benchmark is intentionally warm-path only.
    with httpx.Client(timeout=5.0, follow_redirects=False) as sync_client:
        tags = sync_client.get(f"{args.ollama_url.rstrip('/')}/api/tags")
        tags.raise_for_status()
        ps = sync_client.get(f"{args.ollama_url.rstrip('/')}/api/ps")
        ps.raise_for_status()
        resident_names = {
            str(item.get("name") or item.get("model") or "")
            for item in ps.json().get("models", [])
            if isinstance(item, dict)
        }
    if args.model not in resident_names and f"{args.model}:latest" not in resident_names:
        raise RuntimeError(
            f"{args.model} is not resident. Pre-warm it before running this warm-path benchmark."
        )

    current_path: list[dict[str, Any]] = []
    shared_path: list[dict[str, Any]] = []

    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as shared_client:
        # Alternate order to reduce timing/order bias.
        for index in range(repeats):
            order = ["current", "shared"] if index % 2 == 0 else ["shared", "current"]
            for target in order:
                if target == "current":
                    result = await _measure_preflight(
                        endpoint=args.ollama_url,
                        model=args.model,
                        shared_client=None,
                    )
                    current_path.append(result)
                else:
                    result = await _measure_preflight(
                        endpoint=args.ollama_url,
                        model=args.model,
                        shared_client=shared_client,
                    )
                    shared_path.append(result)

    def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in (
            "provider_construct_seconds",
            "residency_prepare_seconds",
            "scheduler_admission_seconds",
            "preflight_total_seconds",
        ):
            values = [float(run[key]) for run in runs if isinstance(run.get(key), (int, float))]
            result[f"median_{key}"] = _median(values)
        return result

    current_summary = summarize(current_path)
    shared_summary = summarize(shared_path)
    current_total = current_summary.get("median_preflight_total_seconds")
    shared_total = shared_summary.get("median_preflight_total_seconds")
    savings = (
        float(current_total) - float(shared_total)
        if isinstance(current_total, (int, float)) and isinstance(shared_total, (int, float))
        else None
    )
    ratio = (
        float(shared_total) / float(current_total)
        if isinstance(current_total, (int, float))
        and isinstance(shared_total, (int, float))
        and current_total > 0
        else None
    )

    return {
        "benchmark_design": "provider-preflight-fresh-vs-shared-http-client-v1",
        "model": args.model,
        "ollama_url": args.ollama_url,
        "repeats": repeats,
        "current_fresh_client_runs": current_path,
        "shared_client_runs": shared_path,
        "summary": {
            "current_fresh_client": current_summary,
            "shared_client": shared_summary,
            "median_shared_client_savings_seconds": round(savings, 6) if savings is not None else None,
            "shared_vs_current_preflight_ratio": round(ratio, 4) if ratio is not None else None,
        },
        "interpretation_note": (
            "The current /api/chat path constructs a new OllamaProvider without an injected AsyncClient. "
            "That makes the residency /api/ps check use a short-lived HTTP client. The shared-client arm "
            "uses one AsyncClient across rounds while keeping the same residency and scheduler policy."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure SovereignAI provider preflight cost with fresh versus shared HTTP clients."
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("workspace/provider_preflight.json"))
    args = parser.parse_args()

    report = asyncio.run(_run(args))
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
