#!/usr/bin/env python3
"""Compare General vs Authorized local-generation latency and runtime metrics."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import httpx


GENERAL_PROMPT = "In three concise bullet points, explain why preventive maintenance matters."
AUTHORIZED_PROMPT = (
    "According to the authorized knowledge base, summarize the maintenance guidance for Pump-102. "
    "Cite concrete evidence identifiers and abstain from unsupported claims."
)
TERMINAL_EVENTS = {"task_completed", "task_failed", "task_cancelled", "access_revoked"}


def _parse_sse(response: httpx.Response) -> Iterable[tuple[str | None, dict[str, Any]]]:
    event_name: str | None = None
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line == "":
            if event_name is not None or data_lines:
                raw = "\n".join(data_lines)
                try:
                    decoded = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    decoded = {"raw": raw}
                yield event_name, decoded if isinstance(decoded, dict) else {"data": decoded}
            event_name = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())


def _payload(envelope: dict[str, Any]) -> dict[str, Any]:
    nested = envelope.get("payload")
    return nested if isinstance(nested, dict) else envelope


def run_task(client: httpx.Client, *, prompt: str, chat_mode: str, iteration: int) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.post(
        "/api/tasks/start",
        json={
            "request": prompt,
            "chat_mode": chat_mode,
            "execution_mode": "STANDARD",
            "attachments": [],
        },
    )
    response.raise_for_status()
    task_id = str(response.json()["task_id"])
    first_token: float | None = None
    runtime: dict[str, Any] = {}
    terminal: str | None = None
    with client.stream("GET", f"/api/tasks/{task_id}/events", headers={"Accept": "text/event-stream"}) as stream:
        stream.raise_for_status()
        for event_name, envelope in _parse_sse(stream):
            kind = event_name or str(envelope.get("type") or "")
            payload = _payload(envelope)
            if kind == "model_token" and first_token is None:
                first_token = time.perf_counter()
            if kind == "generation_completed" and isinstance(payload.get("runtime_metrics"), dict):
                runtime = dict(payload["runtime_metrics"])
            if kind in TERMINAL_EVENTS:
                terminal = kind
                break
    elapsed = time.perf_counter() - started
    return {
        "chat_mode": chat_mode,
        "iteration": iteration,
        "success": terminal == "task_completed",
        "total_seconds": round(elapsed, 4),
        "client_ttft_seconds": round(first_token - started, 4) if first_token else None,
        "terminal": terminal,
        "runtime": runtime,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row["runtime"][key]) for row in rows if isinstance(row.get("runtime", {}).get(key), (int, float))]
    return round(statistics.fmean(values), 4) if values else None


def summarize(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    selected = [row for row in rows if row["chat_mode"] == mode and row["success"]]
    totals = [row["total_seconds"] for row in selected]
    ttft = [row["client_ttft_seconds"] for row in selected if row["client_ttft_seconds"] is not None]
    return {
        "successes": len(selected),
        "total_mean_s": round(statistics.fmean(totals), 4) if totals else None,
        "ttft_mean_s": round(statistics.fmean(ttft), 4) if ttft else None,
        "tokens_per_second_mean": _mean(selected, "tokens_per_second"),
        "output_tokens_mean": _mean(selected, "token_count"),
        "prompt_tokens_mean": _mean(selected, "prompt_token_count"),
        "load_duration_mean_s": _mean(selected, "load_duration_seconds"),
        "queue_wait_mean_s": _mean(selected, "queue_wait_seconds"),
        "model_duration_mean_s": _mean(selected, "total_duration_seconds"),
        "model_prompt_characters_mean": _mean(selected, "model_prompt_characters"),
        "requested_num_ctx": selected[-1]["runtime"].get("requested_num_ctx") if selected else None,
        "requested_num_predict": selected[-1]["runtime"].get("requested_num_predict") if selected else None,
        "prompt_compacted": selected[-1]["runtime"].get("prompt_compacted") if selected else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--output", default="reports/system_benchmark/authorized_latency_comparison.json")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    with httpx.Client(base_url=args.base_url, timeout=httpx.Timeout(args.timeout)) as client:
        for iteration in range(1, args.runs + 1):
            print(f"[{iteration}/{args.runs}] GENERAL")
            rows.append(run_task(client, prompt=GENERAL_PROMPT, chat_mode="GENERAL", iteration=iteration))
            print(f"[{iteration}/{args.runs}] AUTHORIZED")
            rows.append(run_task(client, prompt=AUTHORIZED_PROMPT, chat_mode="AUTHORIZED", iteration=iteration))

    document = {
        "runs": args.runs,
        "summary": {
            "GENERAL": summarize(rows, "GENERAL"),
            "AUTHORIZED": summarize(rows, "AUTHORIZED"),
        },
        "measurements": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document["summary"], indent=2))
    print(f"Report: {output}")
    return 0 if all(row["success"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
