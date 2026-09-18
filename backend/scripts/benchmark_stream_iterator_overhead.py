from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, AsyncIterator
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


async def _plain_lines(response: httpx.Response) -> AsyncIterator[str]:
    async for line in response.aiter_lines():
        yield line


async def _wait_for_lines(response: httpx.Response, total_timeout_seconds: float) -> AsyncIterator[str]:
    """Mirror OllamaProvider._cancellable_lines when cancellation_event is None."""
    iterator = response.aiter_lines().__aiter__()
    deadline = perf_counter() + total_timeout_seconds
    while True:
        if perf_counter() >= deadline:
            raise asyncio.TimeoutError(
                f"Generation exceeded {total_timeout_seconds:g} seconds"
            )
        try:
            yield await asyncio.wait_for(
                anext(iterator),
                max(0.001, deadline - perf_counter()),
            )
        except StopAsyncIteration:
            return


async def _stream_once(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict[str, Any],
    *,
    mode: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = perf_counter()
    first_token_at: float | None = None
    final_item: dict[str, Any] = {}
    pieces: list[str] = []
    nonempty_lines = 0
    response_lines = 0

    async with client.stream("POST", f"{endpoint.rstrip('/')}/api/generate", json=payload) as response:
        response.raise_for_status()
        source = (
            _plain_lines(response)
            if mode == "plain"
            else _wait_for_lines(response, timeout_seconds)
        )
        async for line in source:
            response_lines += 1
            if not line.strip():
                continue
            nonempty_lines += 1
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
    overhead = wall - total_seconds if total_seconds else None

    return {
        "mode": mode,
        "wall_seconds": round(wall, 6),
        "time_to_first_token_seconds": round(first_token_at - started, 6) if first_token_at else None,
        "ollama_total_duration_seconds": round(total_seconds, 6) if total_seconds else None,
        "load_duration_seconds": round(load_seconds, 6) if load_seconds else 0.0,
        "stream_overhead_seconds": round(overhead, 6) if overhead is not None else None,
        "tokens_per_second": round(eval_count / eval_seconds, 3) if eval_seconds and eval_count else None,
        "token_count": eval_count,
        "response_lines": response_lines,
        "nonempty_lines": nonempty_lines,
        "response": "".join(pieces).strip(),
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    _require_local(args.ollama_url)
    repeats = max(4, int(args.repeats))
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
    plain_runs: list[dict[str, Any]] = []
    wait_runs: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        # Exact runner warm-up outside measured rounds.
        warm_payload = dict(payload)
        warm_payload["options"] = dict(options)
        warm_payload["options"]["num_predict"] = 1
        warm = await _stream_once(
            client,
            args.ollama_url,
            warm_payload,
            mode="plain",
            timeout_seconds=args.timeout,
        )

        for index in range(repeats):
            order = ["plain", "wait_for"] if index % 2 == 0 else ["wait_for", "plain"]
            for mode in order:
                result = await _stream_once(
                    client,
                    args.ollama_url,
                    payload,
                    mode=mode,
                    timeout_seconds=args.timeout,
                )
                if mode == "plain":
                    plain_runs.append(result)
                else:
                    wait_runs.append(result)

    def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in (
            "wall_seconds",
            "time_to_first_token_seconds",
            "ollama_total_duration_seconds",
            "load_duration_seconds",
            "stream_overhead_seconds",
            "tokens_per_second",
            "response_lines",
            "nonempty_lines",
        ):
            values = [float(run[key]) for run in runs if isinstance(run.get(key), (int, float))]
            out[f"median_{key}"] = _median(values)
        return out

    plain_summary = summarize(plain_runs)
    wait_summary = summarize(wait_runs)
    plain_overhead = plain_summary.get("median_stream_overhead_seconds")
    wait_overhead = wait_summary.get("median_stream_overhead_seconds")
    extra = (
        float(wait_overhead) - float(plain_overhead)
        if isinstance(wait_overhead, (int, float)) and isinstance(plain_overhead, (int, float))
        else None
    )

    plain_wall = plain_summary.get("median_wall_seconds")
    wait_wall = wait_summary.get("median_wall_seconds")
    wall_extra = (
        float(wait_wall) - float(plain_wall)
        if isinstance(wait_wall, (int, float)) and isinstance(plain_wall, (int, float))
        else None
    )

    return {
        "benchmark_design": "plain-aiter-lines-vs-per-line-asyncio-wait-for-v1",
        "model": args.model,
        "ollama_url": args.ollama_url,
        "repeats": repeats,
        "options": options,
        "warmup": warm,
        "plain_runs": plain_runs,
        "wait_for_runs": wait_runs,
        "summary": {
            "plain": plain_summary,
            "wait_for": wait_summary,
            "median_wait_for_extra_stream_overhead_seconds": round(extra, 6) if extra is not None else None,
            "median_wait_for_extra_wall_seconds": round(wall_extra, 6) if wall_extra is not None else None,
        },
        "interpretation_note": (
            "The wait_for arm mirrors OllamaProvider._cancellable_lines when no cancellation event is supplied: "
            "each response line is awaited through asyncio.wait_for with the remaining total timeout. The plain "
            "arm consumes response.aiter_lines() directly. Both use the same persistent AsyncClient, model, "
            "prompt and runtime options, and order alternates between rounds."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure per-line asyncio.wait_for overhead in the Ollama streaming path."
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--keep-alive", default="15m")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--num-predict", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-thread", type=int, default=5)
    parser.add_argument("--num-batch", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("workspace/stream_iterator_overhead.json"))
    args = parser.parse_args()

    report = asyncio.run(_run(args))
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
