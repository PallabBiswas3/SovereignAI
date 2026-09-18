from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import httpx

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

LOCALHOST_URL = "http://localhost:11434"
IPV4_URL = "http://127.0.0.1:11434"
SYSTEM_PROMPT = "You are SovereignAI, a local enterprise assistant. Be concise and never invent sources."
PROMPT = "Reply with exactly OK."


def _median(values: list[float]) -> float | None:
    return round(median(values), 6) if values else None


async def _fresh_ps(endpoint: str, timeout: float) -> dict[str, Any]:
    lifecycle_started = perf_counter()
    construct_started = perf_counter()
    client = httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=False)
    construct_seconds = perf_counter() - construct_started
    request_started = perf_counter()
    try:
        response = await client.get(f"{endpoint}/api/ps")
        request_seconds = perf_counter() - request_started
        response.raise_for_status()
        model_count = len(response.json().get("models", []))
    finally:
        close_started = perf_counter()
        await client.aclose()
        close_seconds = perf_counter() - close_started
    lifecycle_seconds = perf_counter() - lifecycle_started
    return {
        "endpoint": endpoint,
        "construct_seconds": round(construct_seconds, 6),
        "request_seconds": round(request_seconds, 6),
        "close_seconds": round(close_seconds, 6),
        "lifecycle_seconds": round(lifecycle_seconds, 6),
        "model_count": model_count,
    }


async def _fresh_generate(endpoint: str, model: str, timeout: float, keep_alive: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": PROMPT,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "options": {
            "num_ctx": 2048,
            "num_predict": 1,
            "temperature": 0.0,
            "num_thread": 5,
            "num_batch": 128,
        },
    }
    lifecycle_started = perf_counter()
    construct_started = perf_counter()
    client = httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=False)
    construct_seconds = perf_counter() - construct_started
    request_started = perf_counter()
    try:
        response = await client.post(f"{endpoint}/api/generate", json=payload)
        request_seconds = perf_counter() - request_started
        response.raise_for_status()
        data = response.json()
    finally:
        close_started = perf_counter()
        await client.aclose()
        close_seconds = perf_counter() - close_started
    lifecycle_seconds = perf_counter() - lifecycle_started
    ollama_total = float(data.get("total_duration", 0) or 0) / 1_000_000_000
    load_seconds = float(data.get("load_duration", 0) or 0) / 1_000_000_000
    return {
        "endpoint": endpoint,
        "construct_seconds": round(construct_seconds, 6),
        "request_seconds": round(request_seconds, 6),
        "close_seconds": round(close_seconds, 6),
        "lifecycle_seconds": round(lifecycle_seconds, 6),
        "ollama_total_duration_seconds": round(ollama_total, 6),
        "request_minus_ollama_seconds": round(request_seconds - ollama_total, 6),
        "load_duration_seconds": round(load_seconds, 6),
        "response": str(data.get("response") or "").strip(),
    }


async def _warm(model: str, timeout: float, keep_alive: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=False) as client:
        payload = {
            "model": model,
            "prompt": "/no_think\nReply with exactly READY.",
            "stream": False,
            "think": False,
            "keep_alive": keep_alive,
            "options": {
                "num_ctx": 2048,
                "num_predict": 1,
                "temperature": 0.0,
                "num_thread": 5,
                "num_batch": 128,
            },
        }
        started = perf_counter()
        response = await client.post(f"{IPV4_URL}/api/generate", json=payload)
        wall = perf_counter() - started
        response.raise_for_status()
        data = response.json()
        return {
            "wall_seconds": round(wall, 6),
            "load_duration_seconds": round(float(data.get("load_duration", 0) or 0) / 1_000_000_000, 6),
        }


def _summarize(runs: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        values = [float(run[key]) for run in runs if isinstance(run.get(key), (int, float))]
        result[f"median_{key}"] = _median(values)
    return result


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    repeats = max(6, args.repeats)
    warmup = await _warm(args.model, args.timeout, args.keep_alive)

    ps_runs: dict[str, list[dict[str, Any]]] = {"localhost": [], "ipv4": []}
    generate_runs: dict[str, list[dict[str, Any]]] = {"localhost": [], "ipv4": []}

    for index in range(repeats):
        order = ["localhost", "ipv4"] if index % 2 == 0 else ["ipv4", "localhost"]
        for label in order:
            endpoint = LOCALHOST_URL if label == "localhost" else IPV4_URL
            ps_runs[label].append(await _fresh_ps(endpoint, args.timeout))
        for label in order:
            endpoint = LOCALHOST_URL if label == "localhost" else IPV4_URL
            generate_runs[label].append(
                await _fresh_generate(endpoint, args.model, args.timeout, args.keep_alive)
            )

    ps_keys = ("construct_seconds", "request_seconds", "close_seconds", "lifecycle_seconds")
    generation_keys = (
        "construct_seconds",
        "request_seconds",
        "close_seconds",
        "lifecycle_seconds",
        "ollama_total_duration_seconds",
        "request_minus_ollama_seconds",
        "load_duration_seconds",
    )
    localhost_ps = _summarize(ps_runs["localhost"], ps_keys)
    ipv4_ps = _summarize(ps_runs["ipv4"], ps_keys)
    localhost_gen = _summarize(generate_runs["localhost"], generation_keys)
    ipv4_gen = _summarize(generate_runs["ipv4"], generation_keys)

    ps_delta = (
        float(localhost_ps["median_lifecycle_seconds"]) - float(ipv4_ps["median_lifecycle_seconds"])
    )
    generation_delta = (
        float(localhost_gen["median_lifecycle_seconds"]) - float(ipv4_gen["median_lifecycle_seconds"])
    )
    return {
        "benchmark_design": "ollama-localhost-vs-ipv4-loopback-v1",
        "model": args.model,
        "repeats": repeats,
        "warmup": warmup,
        "ps_runs": ps_runs,
        "generate_runs": generate_runs,
        "summary": {
            "localhost_ps": localhost_ps,
            "ipv4_ps": ipv4_ps,
            "localhost_generate": localhost_gen,
            "ipv4_generate": ipv4_gen,
            "median_localhost_minus_ipv4_ps_lifecycle_seconds": round(ps_delta, 6),
            "median_localhost_minus_ipv4_generate_lifecycle_seconds": round(generation_delta, 6),
        },
        "interpretation_note": (
            "SovereignAI's config/models.yaml currently uses http://localhost:11434 while the standalone Phase 11 "
            "benchmarks use http://127.0.0.1:11434. This benchmark alternates both aliases against the same warm "
            "Ollama server using fresh AsyncClients and one-token generation to isolate hostname/loopback overhead."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Ollama localhost and IPv4 loopback latency.")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--keep-alive", default="15m")
    parser.add_argument("--output", type=Path, default=Path("workspace/loopback_hostname.json"))
    args = parser.parse_args()

    report = asyncio.run(_run(args))
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
