#!/usr/bin/env python3
"""Focused benchmark for Authorized/RAG evidence distillation.

Run this script after starting the SovereignAI stack with one fixed distillation
configuration. It measures the real user-visible AUTHORIZED path and captures
Ollama prompt/decode metrics from the SSE stream plus the persisted model-facing
sources returned in the terminal task state.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


DEFAULT_PROMPT = (
    "According to the authorized knowledge base, summarize the maintenance guidance for Pump-102. "
    "Cite concrete evidence identifiers and abstain from unsupported claims."
)
TERMINAL_EVENTS = {"task_completed", "task_failed", "task_cancelled", "access_revoked"}


@dataclass
class HostSnapshot:
    cpu_percent: float | None = None
    memory_used_mb: float | None = None
    memory_available_mb: float | None = None
    memory_percent: float | None = None


@dataclass
class RagMeasurement:
    iteration: int
    success: bool
    total_seconds: float
    ttft_seconds: float | None
    prompt_token_count: int | None
    output_token_count: int | None
    tokens_per_second: float | None
    done_reason: str | None
    output_truncated: bool | None
    warm_status: str | None
    selected_source_count: int
    selected_evidence_tokens_estimate: int
    response_chars: int
    citation_identifier_count: int
    warning_count: int
    governance_decision: str | None
    host_before: HostSnapshot
    host_after: HostSnapshot
    error: str | None = None


def host_snapshot() -> HostSnapshot:
    if psutil is None:
        return HostSnapshot()
    memory = psutil.virtual_memory()
    return HostSnapshot(
        cpu_percent=round(float(psutil.cpu_percent(interval=None)), 2),
        memory_used_mb=round(float(memory.used) / (1024 * 1024), 2),
        memory_available_mb=round(float(memory.available) / (1024 * 1024), 2),
        memory_percent=round(float(memory.percent), 2),
    )


def estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\w+|[^\w\s]", text))) if text else 0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def parse_sse(response: httpx.Response) -> Iterable[tuple[str | None, dict[str, Any] | None]]:
    event_name: str | None = None
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line == "":
            if event_name is not None or data_lines:
                payload: dict[str, Any] | None = None
                if data_lines:
                    raw = "\n".join(data_lines)
                    try:
                        decoded = json.loads(raw)
                        payload = decoded if isinstance(decoded, dict) else {"data": decoded}
                    except json.JSONDecodeError:
                        payload = {"raw": raw}
                yield event_name, payload
            event_name, data_lines = None, []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if event_name is not None or data_lines:
        raw = "\n".join(data_lines)
        payload = None
        if raw:
            try:
                decoded = json.loads(raw)
                payload = decoded if isinstance(decoded, dict) else {"data": decoded}
            except json.JSONDecodeError:
                payload = {"raw": raw}
        yield event_name, payload


def event_payload(envelope: dict[str, Any] | None) -> dict[str, Any]:
    if not envelope:
        return {}
    nested = envelope.get("payload")
    return nested if isinstance(nested, dict) else envelope


def event_type(name: str | None, envelope: dict[str, Any] | None) -> str | None:
    if name:
        return name
    if envelope and envelope.get("type"):
        return str(envelope["type"])
    return None


def _fmt_metric(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def run_once(
    client: httpx.Client,
    *,
    prompt: str,
    iteration: int,
    total_runs: int,
    label: str,
    timeout_seconds: float,
) -> RagMeasurement:
    host_before = host_snapshot()
    started = time.perf_counter()
    first_token_at: float | None = None
    runtime: dict[str, Any] = {}
    terminal_type: str | None = None
    terminal_payload: dict[str, Any] = {}

    print(f"\n[{label}] Run {iteration}/{total_runs} started...", flush=True)
    if host_before.memory_percent is not None:
        print(
            f"[{label}] Host before: RAM {host_before.memory_percent:.1f}% "
            f"({host_before.memory_available_mb:.0f} MB available)",
            flush=True,
        )

    try:
        start = client.post(
            "/api/tasks/start",
            json={
                "request": prompt,
                "chat_mode": "AUTHORIZED",
                "execution_mode": "STANDARD",
                "attachments": [],
            },
        )
        start.raise_for_status()
        task_id = str(start.json()["task_id"])
        print(f"[{label}] Task accepted: {task_id}", flush=True)
        print(f"[{label}] Waiting for retrieval/prefill and first model token...", flush=True)

        with client.stream(
            "GET",
            f"/api/tasks/{task_id}/events",
            headers={"Accept": "text/event-stream"},
            timeout=httpx.Timeout(timeout_seconds),
        ) as response:
            response.raise_for_status()
            for name, envelope in parse_sse(response):
                kind = event_type(name, envelope)
                payload = event_payload(envelope)
                if kind == "model_token" and first_token_at is None:
                    first_token_at = time.perf_counter()
                    print(
                        f"[{label}] First token received after "
                        f"{first_token_at - started:.2f} s; generation is in progress...",
                        flush=True,
                    )
                elif kind == "generation_completed":
                    metrics = payload.get("runtime_metrics")
                    if isinstance(metrics, dict):
                        runtime = metrics
                        print(
                            f"[{label}] Model generation completed: "
                            f"prompt={_fmt_metric(runtime.get('prompt_token_count'))} tokens, "
                            f"output={_fmt_metric(runtime.get('token_count'))} tokens, "
                            f"done_reason={_fmt_metric(runtime.get('done_reason'))}",
                            flush=True,
                        )
                if kind in TERMINAL_EVENTS:
                    terminal_type = kind
                    terminal_payload = payload
                    print(f"[{label}] Terminal event: {kind}", flush=True)
                    break

        elapsed = time.perf_counter() - started
        result = terminal_payload.get("result") if isinstance(terminal_payload, dict) else None
        result = result if isinstance(result, dict) else {}
        sources = result.get("sources") if isinstance(result.get("sources"), list) else []
        evidence_text = "\n".join(
            str(item.get("text") or "") for item in sources if isinstance(item, dict)
        )
        final_response = str(result.get("final_response") or "")
        warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []
        governance = result.get("governance") if isinstance(result.get("governance"), dict) else {}
        identifiers = set(
            re.findall(
                r"\b(?:E\d+|F\d+|[A-Z]{2,}(?:-[A-Z0-9]+)+|[0-9a-f]{8}-[0-9a-f-]{27,})\b",
                final_response,
                re.I,
            )
        )
        success = terminal_type == "task_completed"
        measurement = RagMeasurement(
            iteration=iteration,
            success=success,
            total_seconds=round(elapsed, 6),
            ttft_seconds=round(first_token_at - started, 6) if first_token_at else None,
            prompt_token_count=_int_or_none(runtime.get("prompt_token_count")),
            output_token_count=_int_or_none(runtime.get("token_count")),
            tokens_per_second=_float_or_none(runtime.get("tokens_per_second")),
            done_reason=_str_or_none(runtime.get("done_reason")),
            output_truncated=(bool(runtime.get("output_truncated")) if "output_truncated" in runtime else None),
            warm_status=_str_or_none(runtime.get("warm_status")),
            selected_source_count=len(sources),
            selected_evidence_tokens_estimate=estimate_tokens(evidence_text),
            response_chars=len(final_response),
            citation_identifier_count=len(identifiers),
            warning_count=len(warnings),
            governance_decision=_str_or_none(governance.get("decision")),
            host_before=host_before,
            host_after=host_snapshot(),
            error=None if success else json.dumps(terminal_payload, default=str)[:1000],
        )
        print(
            f"[{label}] Run {iteration}/{total_runs} {'completed' if success else 'failed'} | "
            f"TTFT={_fmt_metric(measurement.ttft_seconds, ' s')} | "
            f"total={_fmt_metric(measurement.total_seconds, ' s')} | "
            f"prompt={_fmt_metric(measurement.prompt_token_count)} | "
            f"evidence~={measurement.selected_evidence_tokens_estimate} | "
            f"output={_fmt_metric(measurement.output_token_count)} | "
            f"tok/s={_fmt_metric(measurement.tokens_per_second)} | "
            f"truncated={measurement.output_truncated} | "
            f"sources={measurement.selected_source_count} | "
            f"governance={measurement.governance_decision or 'n/a'}",
            flush=True,
        )
        return measurement
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        measurement = RagMeasurement(
            iteration=iteration,
            success=False,
            total_seconds=round(time.perf_counter() - started, 6),
            ttft_seconds=None,
            prompt_token_count=None,
            output_token_count=None,
            tokens_per_second=None,
            done_reason=None,
            output_truncated=None,
            warm_status=None,
            selected_source_count=0,
            selected_evidence_tokens_estimate=0,
            response_chars=0,
            citation_identifier_count=0,
            warning_count=0,
            governance_decision=None,
            host_before=host_before,
            host_after=host_snapshot(),
            error=str(exc),
        )
        print(
            f"[{label}] Run {iteration}/{total_runs} failed after "
            f"{measurement.total_seconds:.2f} s: {exc}",
            flush=True,
        )
        return measurement


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _str_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def summarize(rows: list[RagMeasurement]) -> dict[str, Any]:
    successful = [row for row in rows if row.success]

    def nums(attribute: str) -> list[float]:
        output: list[float] = []
        for row in successful:
            value = getattr(row, attribute)
            if isinstance(value, (int, float)):
                output.append(float(value))
        return output

    totals = nums("total_seconds")
    ttfts = nums("ttft_seconds")
    prompts = nums("prompt_token_count")
    evidence = nums("selected_evidence_tokens_estimate")
    outputs = nums("output_token_count")
    return {
        "runs": len(rows),
        "successes": len(successful),
        "success_rate": round(len(successful) / max(1, len(rows)), 4),
        "total_p50_s": _round(percentile(totals, 0.50)),
        "total_p95_s": _round(percentile(totals, 0.95)),
        "ttft_p50_s": _round(percentile(ttfts, 0.50)),
        "ttft_p95_s": _round(percentile(ttfts, 0.95)),
        "prompt_tokens_mean": _mean(prompts),
        "evidence_tokens_estimate_mean": _mean(evidence),
        "output_tokens_mean": _mean(outputs),
        "tokens_per_second_mean": _mean(nums("tokens_per_second"), 3),
        "truncated_runs": sum(row.output_truncated is True for row in successful),
        "citation_identifier_count_mean": _mean(nums("citation_identifier_count")),
        "governance_decisions": sorted({row.governance_decision for row in successful if row.governance_decision}),
    }


def _mean(values: list[float], digits: int = 2) -> float | None:
    return round(statistics.fmean(values), digits) if values else None


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Authorized/RAG evidence distillation.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--label", required=True, help="Example: control, 900, 700, or 500")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/rag_distillation"))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    client = httpx.Client(base_url=args.base_url.rstrip("/"), timeout=httpx.Timeout(args.timeout))
    rows: list[RagMeasurement] = []
    try:
        print(f"[{args.label}] Checking SovereignAI health...", flush=True)
        health = client.get("/health")
        health.raise_for_status()
        print(
            f"[{args.label}] Health OK. Starting {args.runs} Authorized/RAG benchmark run(s).",
            flush=True,
        )
        for iteration in range(1, args.runs + 1):
            rows.append(
                run_once(
                    client,
                    prompt=args.prompt,
                    iteration=iteration,
                    total_runs=args.runs,
                    label=args.label,
                    timeout_seconds=args.timeout,
                )
            )
    finally:
        client.close()

    document = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "label": args.label,
            "base_url": args.base_url,
            "runs": args.runs,
            "prompt": args.prompt,
        },
        "summary": summarize(rows),
        "measurements": [asdict(row) for row in rows],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"rag_distillation_{args.label}.json"
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[{args.label}] All runs finished. Summary:", flush=True)
    print(json.dumps({"summary": document["summary"], "report": str(path)}, indent=2), flush=True)
    return 0 if all(row.success for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())