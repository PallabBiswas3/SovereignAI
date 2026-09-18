from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

import httpx

DEFAULT_PROMPT = (
    "Answer with exactly four short bullets, one sentence each: why does an outer-race "
    "bearing fault create periodic vibration impulses?"
)
TERMINAL_EVENTS = {"task_completed", "task_failed", "task_cancelled", "access_revoked"}


def _require_local(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"Benchmark endpoint must be loopback-local: {url}")


def _median(values: list[float]) -> float | None:
    return round(median(values), 6) if values else None


def _iso_to_epoch(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


async def _read_sse(
    response: httpx.Response,
    *,
    benchmark_started: float,
    benchmark_started_epoch: float,
    ack_at: float,
    sse_headers_at: float,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    current_event_name: str | None = None
    data_lines: list[str] = []
    first_event_at: float | None = None
    first_model_token_at: float | None = None
    terminal_at: float | None = None
    terminal_type: str | None = None
    model_token_receive_times: list[float] = []
    model_token_created_to_receive: list[float] = []
    event_created_to_receive: list[float] = []
    generation_started_receive_at: float | None = None
    generation_completed_receive_at: float | None = None

    async def consume_frame() -> bool:
        nonlocal current_event_name, data_lines
        nonlocal first_event_at, first_model_token_at, terminal_at, terminal_type
        nonlocal generation_started_receive_at, generation_completed_receive_at
        if not data_lines:
            current_event_name = None
            return False
        received_at = perf_counter()
        received_epoch = benchmark_started_epoch + (received_at - benchmark_started)
        raw = "\n".join(data_lines)
        data_lines = []
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            current_event_name = None
            return False
        if not isinstance(event, dict):
            current_event_name = None
            return False
        event_type = str(event.get("type") or current_event_name or "message")
        created_epoch = _iso_to_epoch(event.get("timestamp"))
        created_to_receive = (
            max(0.0, received_epoch - created_epoch) if created_epoch is not None else None
        )
        record = {
            "type": event_type,
            "timestamp": event.get("timestamp"),
            "receive_offset_seconds": round(received_at - benchmark_started, 6),
            "ack_to_receive_seconds": round(received_at - ack_at, 6),
            "sse_headers_to_receive_seconds": round(received_at - sse_headers_at, 6),
            "event_created_to_receive_seconds": round(created_to_receive, 6)
            if created_to_receive is not None
            else None,
            "payload": event.get("payload") if isinstance(event.get("payload"), dict) else {},
        }
        events.append(record)
        if created_to_receive is not None:
            event_created_to_receive.append(created_to_receive)
        if first_event_at is None:
            first_event_at = received_at
        if event_type == "generation_started" and generation_started_receive_at is None:
            generation_started_receive_at = received_at
        if event_type == "model_token":
            if first_model_token_at is None:
                first_model_token_at = received_at
            model_token_receive_times.append(received_at)
            if created_to_receive is not None:
                model_token_created_to_receive.append(created_to_receive)
        if event_type == "generation_completed" and generation_completed_receive_at is None:
            generation_completed_receive_at = received_at
        if event_type in TERMINAL_EVENTS:
            terminal_at = received_at
            terminal_type = event_type
            current_event_name = None
            return True
        current_event_name = None
        return False

    async for line in response.aiter_lines():
        if line == "":
            if await consume_frame():
                break
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            current_event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines and terminal_at is None:
        await consume_frame()

    type_counts = Counter(str(item["type"]) for item in events)
    first_offsets: dict[str, float] = {}
    for item in events:
        event_type = str(item["type"])
        first_offsets.setdefault(event_type, float(item["receive_offset_seconds"]))

    token_intervals = [
        model_token_receive_times[index] - model_token_receive_times[index - 1]
        for index in range(1, len(model_token_receive_times))
    ]
    terminal_payload: dict[str, Any] = {}
    if events and terminal_type:
        for item in reversed(events):
            if item["type"] == terminal_type:
                terminal_payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
                break
    completed_result = terminal_payload.get("result") if isinstance(terminal_payload.get("result"), dict) else {}
    runtime_metrics = completed_result.get("runtime_metrics") if isinstance(completed_result.get("runtime_metrics"), dict) else {}
    latency_breakdown = runtime_metrics.get("latency_breakdown") if isinstance(runtime_metrics.get("latency_breakdown"), dict) else {}

    return {
        "event_count": len(events),
        "event_type_counts": dict(sorted(type_counts.items())),
        "first_event_type": events[0]["type"] if events else None,
        "terminal_event_type": terminal_type,
        "first_event_offset_seconds": round(first_event_at - benchmark_started, 6)
        if first_event_at is not None
        else None,
        "ack_to_first_event_seconds": round(first_event_at - ack_at, 6)
        if first_event_at is not None
        else None,
        "first_model_token_offset_seconds": round(first_model_token_at - benchmark_started, 6)
        if first_model_token_at is not None
        else None,
        "ack_to_first_model_token_seconds": round(first_model_token_at - ack_at, 6)
        if first_model_token_at is not None
        else None,
        "sse_headers_to_first_model_token_seconds": round(first_model_token_at - sse_headers_at, 6)
        if first_model_token_at is not None
        else None,
        "generation_started_to_first_model_token_receive_seconds": round(
            first_model_token_at - generation_started_receive_at, 6
        )
        if first_model_token_at is not None and generation_started_receive_at is not None
        else None,
        "first_model_token_to_generation_completed_receive_seconds": round(
            generation_completed_receive_at - first_model_token_at, 6
        )
        if first_model_token_at is not None and generation_completed_receive_at is not None
        else None,
        "task_total_seconds": round(terminal_at - benchmark_started, 6)
        if terminal_at is not None
        else None,
        "ack_to_terminal_seconds": round(terminal_at - ack_at, 6)
        if terminal_at is not None
        else None,
        "model_token_event_count": len(model_token_receive_times),
        "model_token_interarrival_median_seconds": _median(token_intervals),
        "model_token_interarrival_p95_seconds": _percentile(token_intervals, 0.95),
        "event_created_to_sse_receive_median_seconds": _median(event_created_to_receive),
        "event_created_to_sse_receive_p95_seconds": _percentile(event_created_to_receive, 0.95),
        "model_token_created_to_sse_receive_median_seconds": _median(model_token_created_to_receive),
        "model_token_created_to_sse_receive_p95_seconds": _percentile(model_token_created_to_receive, 0.95),
        "first_event_offsets_by_type_seconds": first_offsets,
        "agent_runtime_metrics": runtime_metrics,
        "agent_latency_breakdown": latency_breakdown,
        "completed_run_id": completed_result.get("id") if isinstance(completed_result, dict) else None,
        "events": events,
    }


async def _run_task(
    client: httpx.AsyncClient,
    *,
    api_url: str,
    prompt: str,
    timeout_seconds: float,
    measured: bool,
) -> dict[str, Any]:
    benchmark_started = perf_counter()
    benchmark_started_epoch = datetime.now(timezone.utc).timestamp()
    body = {
        "request": prompt,
        "model_override": None,
        "use_case": "internal_assistant",
        "attachments": [],
        "execution_mode": "FAST",
        "chat_mode": "GENERAL",
        "workcell_id": None,
    }
    response = await client.post(
        f"{api_url.rstrip('/')}/api/tasks/start",
        json=body,
        timeout=timeout_seconds,
    )
    ack_at = perf_counter()
    if not response.is_success:
        raise RuntimeError(
            f"POST /api/tasks/start returned HTTP {response.status_code}: {response.text[:2000]}"
        )
    started = response.json()
    task_id = str(started.get("task_id") or "")
    if not task_id:
        raise RuntimeError("POST /api/tasks/start did not return task_id")

    stream_started = perf_counter()
    async with client.stream(
        "GET",
        f"{api_url.rstrip('/')}/api/tasks/{task_id}/events",
        timeout=httpx.Timeout(timeout_seconds),
        headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"},
    ) as stream_response:
        sse_headers_at = perf_counter()
        if not stream_response.is_success:
            body_text = await stream_response.aread()
            raise RuntimeError(
                f"GET task SSE returned HTTP {stream_response.status_code}: {body_text[:2000]!r}"
            )
        stream_metrics = await _read_sse(
            stream_response,
            benchmark_started=benchmark_started,
            benchmark_started_epoch=benchmark_started_epoch,
            ack_at=ack_at,
            sse_headers_at=sse_headers_at,
        )

    return {
        "measured": measured,
        "task_id": task_id,
        "start_ack_seconds": round(ack_at - benchmark_started, 6),
        "sse_connect_after_ack_seconds": round(sse_headers_at - ack_at, 6),
        "sse_headers_seconds": round(sse_headers_at - stream_started, 6),
        **stream_metrics,
    }


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "start_ack_seconds",
        "sse_connect_after_ack_seconds",
        "first_event_offset_seconds",
        "ack_to_first_event_seconds",
        "first_model_token_offset_seconds",
        "ack_to_first_model_token_seconds",
        "sse_headers_to_first_model_token_seconds",
        "generation_started_to_first_model_token_receive_seconds",
        "first_model_token_to_generation_completed_receive_seconds",
        "task_total_seconds",
        "ack_to_terminal_seconds",
        "event_created_to_sse_receive_median_seconds",
        "model_token_created_to_sse_receive_median_seconds",
        "model_token_interarrival_median_seconds",
    )
    summary: dict[str, Any] = {}
    for key in keys:
        values = [float(run[key]) for run in runs if isinstance(run.get(key), (int, float))]
        summary[f"median_{key}"] = _median(values)
    token_counts = [float(run["model_token_event_count"]) for run in runs if isinstance(run.get("model_token_event_count"), int)]
    summary["median_model_token_event_count"] = _median(token_counts)

    latency_keys = (
        "generation_wall_seconds",
        "queue_wait_seconds",
        "model_reported_total_seconds",
        "load_duration_seconds",
        "time_to_first_token_seconds",
        "decode_tokens_per_second",
        "first_ui_frame_seconds",
        "event_callback_seconds",
        "event_callback_count",
        "event_callback_percent",
        "application_overhead_seconds",
        "application_overhead_percent",
    )
    for key in latency_keys:
        values = [
            float(run.get("agent_latency_breakdown", {}).get(key))
            for run in runs
            if isinstance(run.get("agent_latency_breakdown"), dict)
            and isinstance(run.get("agent_latency_breakdown", {}).get(key), (int, float))
        ]
        summary[f"median_agent_{key}"] = _median(values)
    return summary


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    _require_local(args.api_url)
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        health = await client.get(f"{args.api_url.rstrip('/')}/health", timeout=5.0)
        if not health.is_success:
            raise RuntimeError(f"SovereignAI backend is not reachable at {args.api_url}")

        warmups: list[dict[str, Any]] = []
        for index in range(max(0, args.warmup_runs)):
            warmups.append(
                await _run_task(
                    client,
                    api_url=args.api_url,
                    prompt=f"{args.prompt}\nWarm-up run {index + 1}.",
                    timeout_seconds=args.timeout,
                    measured=False,
                )
            )

        runs: list[dict[str, Any]] = []
        for index in range(max(1, args.repeats)):
            run = await _run_task(
                client,
                api_url=args.api_url,
                prompt=f"{args.prompt}\nBenchmark run {index + 1}.",
                timeout_seconds=args.timeout,
                measured=True,
            )
            runs.append(run)
            progress = {
                "benchmark_design": "phase12-real-task-sse-latency-v1",
                "status": "running" if index + 1 < max(1, args.repeats) else "complete",
                "api_url": args.api_url,
                "prompt": args.prompt,
                "warmup_runs": warmups,
                "runs": runs,
                "summary": _summarize(runs),
                "semantics": {
                    "live_delivery": (
                        "Task events are published to the in-process broker before TaskEventRecord SQLite commit. "
                        "SSE receive latency therefore measures live broker delivery, while AgentExecutor callback "
                        "time includes the persistence callback cost."
                    ),
                    "browser_paint": (
                        "This v1 benchmark terminates at SSE client receipt. Browser React state update and paint "
                        "are intentionally not claimed yet; they require frontend-side Performance API instrumentation."
                    ),
                },
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(progress, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "benchmark_design": "phase12-real-task-sse-latency-v1",
        "status": "complete",
        "api_url": args.api_url,
        "prompt": args.prompt,
        "requested_repeats": max(1, args.repeats),
        "completed_repeats": len(runs),
        "warmup_runs": warmups,
        "runs": runs,
        "summary": _summarize(runs),
        "semantics": {
            "live_delivery": (
                "Task events are published to the in-process broker before TaskEventRecord SQLite commit. "
                "SSE receive latency therefore measures live broker delivery, while AgentExecutor callback "
                "time includes the persistence callback cost."
            ),
            "browser_paint": (
                "This v1 benchmark terminates at SSE client receipt. Browser React state update and paint "
                "are intentionally not claimed yet; they require frontend-side Performance API instrumentation."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 12 v1: benchmark real task start -> AgentExecutor -> broker -> SSE latency."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=Path("workspace/phase12_task_stream_latency.json"))
    args = parser.parse_args()

    report = asyncio.run(_run(args))
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
