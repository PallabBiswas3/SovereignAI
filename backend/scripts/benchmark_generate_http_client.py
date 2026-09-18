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


async def _stream_once(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    started = perf_counter()
    first_token_at: float | None = None
    final_item: dict[str, Any] = {}
    pieces: list[str] = []
    async with client.stream("POST", f"{endpoint.rstrip('/')}/api/generate", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            item = json.loads(line)
            text = str(item.get("response") or "")
            if text:
                first_token_at = first_token_at or perf_counter()
                pieces.append(text)
            if item.get("done"):
                final_item = item
                break
    wall = perf_counter() - started
    total_seconds = float(final_item.get("total_duration", 0) or 0) / 1_000_000_000
    load_seconds = float(final_item.get("load_duration", 0) or 0) / 1_000_000_000
    eval_seconds = float(final_item.get("eval_duration", 0) or 0) / 1_000_000_000
    eval_count = int(final_item.get("eval_count", 0) or 0)
    transport_overhead = wall - total_seconds if total_seconds else None
    return {
        "wall_seconds": round(wall, 6),
        "time_to_first_token_seconds": round(first_token_at - started, 6) if first_token_at else None,
        "ollama_total_duration_seconds": round(total_seconds, 6) if total_seconds else None,
        "load_duration_seconds": round(load_seconds, 6) if load_seconds else 0.0,
        "tokens_per_second": round(eval_count / eval_seconds, 3) if eval_seconds and eval_count else None,
        "token_count": eval_count,
        "transport_overhead_seconds": round(transport_overhead, 6) if transport_overhead is not None else None,
        "response": "".join(pieces).strip(),
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    _require_local(args.ollama_url)
    repeats = max(4, args.repeats)
    options = {
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "temperature": args.temperature,
        "num_thread": args.num_thread,
        "num_batch": args.num_batch,
    }
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "system": SYSTEM_PROMPT,
        "stream": True,
        "think": False,
        "keep_alive": args.keep_alive,
        "options": options,
    }

    timeout = httpx.Timeout(args.timeout)

    # Exact-runner warm-up outside measured rounds.
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as warm_client:
        warm_payload = dict(payload)
        warm_payload["options"] = dict(options)
        warm_payload["options"]["num_predict"] = 1
        warm = await _stream_once(warm_client, args.ollama_url, warm_payload)

    fresh_runs: list[dict[str, Any]] = []
    shared_runs: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as shared_client:
        for index in range(repeats):
            order = ["fresh", "shared"] if index % 2 == 0 else ["shared", "fresh"]
            for target in order:
                if target == "fresh":
                    construct_started = perf_counter()
                    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as fresh_client:
                        construct_seconds = perf_counter() - construct_started
                        result = await _stream_once(fresh_client, args.ollama_url, payload)
                    result["client_construct_seconds"] = round(construct_seconds, 6)
                    fresh_runs.append(result)
                else:
                    result = await _stream_once(shared_client, args.ollama_url, payload)
                    result["client_construct_seconds"] = 0.0
                    shared_runs.append(result)

    def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in (
            "client_construct_seconds",
            "wall_seconds",
            "time_to_first_token_seconds",
            "ollama_total_duration_seconds",
            "load_duration_seconds",
            "transport_overhead_seconds",
            "tokens_per_second",
        ):
            values = [float(run[key]) for run in runs if isinstance(run.get(key), (int, float))]
            out[f"median_{key}"] = _median(values)
        return out

    fresh_summary = summarize(fresh_runs)
    shared_summary = summarize(shared_runs)
    fresh_wall = fresh_summary.get("median_wall_seconds")
    shared_wall = shared_summary.get("median_wall_seconds")
    savings = (
        float(fresh_wall) - float(shared_wall)
        if isinstance(fresh_wall, (int, float)) and isinstance(shared_wall, (int, float))
        else None
    )
    return {
        "benchmark_design": "generation-fresh-vs-shared-async-http-client-v1",
        "model": args.model,
        "ollama_url": args.ollama_url,
        "repeats": repeats,
        "options": options,
        "warmup": warm,
        "fresh_client_runs": fresh_runs,
        "shared_client_runs": shared_runs,
        "summary": {
            "fresh_client": fresh_summary,
            "shared_client": shared_summary,
            "median_shared_client_wall_savings_seconds": round(savings, 6) if savings is not None else None,
        },
        "interpretation_note": (
            "Both arms call the same already-warm Ollama /api/generate endpoint with the same model and runtime "
            "options. The fresh arm constructs and closes a new AsyncClient for every generation, matching the "
            "current OllamaProvider request pattern. The shared arm reuses one AsyncClient across measured calls."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark fresh versus shared AsyncClient generation overhead.")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--keep-alive", default="15m")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--num-predict", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-thread", type=int, default=5)
    parser.add_argument("--num-batch", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("workspace/generate_http_client.json"))
    args = parser.parse_args()

    report = asyncio.run(_run(args))
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
