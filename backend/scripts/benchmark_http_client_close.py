from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

import httpx

SYSTEM_PROMPT = "You are SovereignAI, a local enterprise assistant. Be concise and never invent sources."
PROMPT = "Answer with exactly four short bullets, one sentence each: why does an outer-race bearing fault create periodic vibration impulses?"


def _require_local(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"Benchmark endpoint must be loopback-local: {url}")


def _median(values: list[float]) -> float | None:
    return round(median(values), 6) if values else None


async def _stream_once(client: httpx.AsyncClient, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    started = perf_counter()
    first_token_at: float | None = None
    final_item: dict[str, Any] = {}
    async with client.stream("POST", f"{endpoint.rstrip('/')}/api/generate", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            item = json.loads(line)
            text = str(item.get("response") or "")
            if text and first_token_at is None:
                first_token_at = perf_counter()
            if item.get("done"):
                final_item = item
                break
    stream_wall = perf_counter() - started
    ollama_total = float(final_item.get("total_duration", 0) or 0) / 1_000_000_000
    load = float(final_item.get("load_duration", 0) or 0) / 1_000_000_000
    return {
        "stream_wall_seconds": round(stream_wall, 6),
        "time_to_first_token_seconds": round(first_token_at - started, 6) if first_token_at else None,
        "ollama_total_duration_seconds": round(ollama_total, 6) if ollama_total else None,
        "load_duration_seconds": round(load, 6) if load else 0.0,
        "stream_minus_ollama_seconds": round(stream_wall - ollama_total, 6) if ollama_total else None,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    _require_local(args.ollama_url)
    payload = {
        "model": args.model,
        "prompt": PROMPT,
        "system": SYSTEM_PROMPT,
        "stream": True,
        "think": False,
        "keep_alive": args.keep_alive,
        "options": {
            "num_ctx": 2048,
            "num_predict": 384,
            "temperature": 0.2,
            "num_thread": 5,
            "num_batch": 128,
        },
    }
    timeout = httpx.Timeout(args.timeout)

    # Exact runner warm-up outside measurement.
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as warm_client:
        warm_payload = dict(payload)
        warm_payload["options"] = dict(payload["options"])
        warm_payload["options"]["num_predict"] = 1
        await _stream_once(warm_client, args.ollama_url, warm_payload)

    runs: list[dict[str, Any]] = []
    for _ in range(max(4, args.repeats)):
        lifecycle_started = perf_counter()

        construct_started = perf_counter()
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)
        construct_seconds = perf_counter() - construct_started

        stream = await _stream_once(client, args.ollama_url, payload)

        close_started = perf_counter()
        await client.aclose()
        close_seconds = perf_counter() - close_started

        lifecycle_seconds = perf_counter() - lifecycle_started
        runs.append({
            **stream,
            "client_construct_seconds": round(construct_seconds, 6),
            "client_close_seconds": round(close_seconds, 6),
            "client_lifecycle_seconds": round(lifecycle_seconds, 6),
            "lifecycle_minus_ollama_seconds": round(
                lifecycle_seconds - float(stream["ollama_total_duration_seconds"] or 0.0), 6
            ),
        })

    summary: dict[str, Any] = {}
    for key in (
        "client_construct_seconds",
        "stream_wall_seconds",
        "client_close_seconds",
        "client_lifecycle_seconds",
        "ollama_total_duration_seconds",
        "stream_minus_ollama_seconds",
        "lifecycle_minus_ollama_seconds",
        "time_to_first_token_seconds",
    ):
        values = [float(run[key]) for run in runs if isinstance(run.get(key), (int, float))]
        summary[f"median_{key}"] = _median(values)

    return {
        "benchmark_design": "ollama-http-client-full-lifecycle-v1",
        "model": args.model,
        "repeats": max(4, args.repeats),
        "runs": runs,
        "summary": summary,
        "interpretation_note": (
            "Unlike the earlier generation benchmark, lifecycle time includes AsyncClient construction, the full "
            "streaming request, and await client.aclose(). This matches the current OllamaProvider ownership path, "
            "where a request-owned client is closed in finally before provider.generate() returns."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure fresh Ollama AsyncClient close/lifecycle overhead.")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--keep-alive", default="15m")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=Path("workspace/http_client_close.json"))
    args = parser.parse_args()

    report = asyncio.run(_run(args))
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
