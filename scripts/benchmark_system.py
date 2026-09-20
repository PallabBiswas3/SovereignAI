#!/usr/bin/env python3
"""Benchmark SovereignAI end-to-end from the same machine that runs the stack.

Measures:
- backend and integration health
- plain /api/chat latency
- streamed task TTFT and total latency through /api/tasks/start + SSE
- authorized/RAG task latency
- GraphRAG + ControlPlane integration latency
- optional GraphRAG + Time-Series Diagnostic Agent + ControlPlane latency
- host CPU/RAM snapshots
- JSON, CSV and Markdown reports

This script is intentionally read-only with respect to external services. It does
not stop services for failure injection. Degraded/unavailable services are
recorded honestly in the report.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx

try:
    import psutil
except ImportError:  # pragma: no cover - backend requirements already include psutil
    psutil = None


TERMINAL_EVENTS = {"task_completed", "task_failed", "task_cancelled", "access_revoked"}
DEFAULT_CHAT_PROMPT = "In three concise bullet points, explain why preventive maintenance matters."
DEFAULT_RAG_PROMPT = (
    "According to the authorized knowledge base, summarize the maintenance guidance for Pump-102. "
    "Cite concrete evidence identifiers and abstain from unsupported claims."
)
DEFAULT_INTEGRATION_QUERY = (
    "Using the available document evidence, summarize the supported maintenance context for Pump-102. "
    "Do not invent facts if evidence is insufficient."
)
DEFAULT_DIAGNOSTIC_QUERY = (
    "Using document evidence and the supplied process time-series evidence, assess whether the pressure "
    "channel shows a process fault. State only evidence-supported conclusions."
)


@dataclass
class HostSnapshot:
    cpu_percent: float | None = None
    memory_used_mb: float | None = None
    memory_available_mb: float | None = None
    memory_percent: float | None = None


@dataclass
class Measurement:
    scenario: str
    iteration: int
    success: bool
    status_code: int | None
    total_seconds: float
    ttft_seconds: float | None = None
    server_model_seconds: float | None = None
    tokens_per_second: float | None = None
    token_count: int | None = None
    warm_status: str | None = None
    retries: int | None = None
    released: bool | None = None
    result_status: str | None = None
    service_status: dict[str, str] = field(default_factory=dict)
    component_timings_ms: dict[str, float] = field(default_factory=dict)
    host_after: HostSnapshot = field(default_factory=HostSnapshot)
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


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_measurements(measurements: Iterable[Measurement]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Measurement]] = {}
    for item in measurements:
        groups.setdefault(item.scenario, []).append(item)
    summary: dict[str, dict[str, Any]] = {}
    for scenario, rows in groups.items():
        totals = [row.total_seconds for row in rows if row.success]
        ttfts = [row.ttft_seconds for row in rows if row.success and row.ttft_seconds is not None]
        tps = [row.tokens_per_second for row in rows if row.success and row.tokens_per_second is not None]
        summary[scenario] = {
            "runs": len(rows),
            "successes": sum(1 for row in rows if row.success),
            "success_rate": round(sum(1 for row in rows if row.success) / len(rows), 4),
            "total_p50_s": round(percentile(totals, 0.50), 4) if totals else None,
            "total_p95_s": round(percentile(totals, 0.95), 4) if totals else None,
            "total_mean_s": round(statistics.fmean(totals), 4) if totals else None,
            "ttft_p50_s": round(percentile(ttfts, 0.50), 4) if ttfts else None,
            "ttft_p95_s": round(percentile(ttfts, 0.95), 4) if ttfts else None,
            "tokens_per_second_mean": round(statistics.fmean(tps), 3) if tps else None,
        }
    return summary


def _parse_sse(response: httpx.Response) -> Iterable[tuple[str | None, dict[str, Any] | None]]:
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
            event_name = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if event_name is not None or data_lines:
        payload = None
        if data_lines:
            raw = "\n".join(data_lines)
            try:
                decoded = json.loads(raw)
                payload = decoded if isinstance(decoded, dict) else {"data": decoded}
            except json.JSONDecodeError:
                payload = {"raw": raw}
        yield event_name, payload


def _event_type(event_name: str | None, payload: dict[str, Any] | None) -> str | None:
    if event_name:
        return event_name
    if payload:
        value = payload.get("type")
        return str(value) if value else None
    return None


def _event_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    nested = payload.get("payload")
    return nested if isinstance(nested, dict) else payload


class BenchmarkRunner:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        bearer_token: str | None,
        cookie: str | None,
        csrf_token: str | None,
    ) -> None:
        headers = {"Accept": "application/json"}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        if cookie:
            headers["Cookie"] = cookie
        if csrf_token:
            headers["X-CSRF-Token"] = csrf_token
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
        )

    def close(self) -> None:
        self.client.close()

    def login(self, email: str, password: str, csrf_cookie_name: str) -> None:
        response = self.client.post("/api/auth/login", json={"email": email, "password": password})
        response.raise_for_status()
        csrf = self.client.cookies.get(csrf_cookie_name)
        if not csrf:
            raise RuntimeError(f"Login succeeded but CSRF cookie {csrf_cookie_name!r} was not set.")
        self.client.headers["X-CSRF-Token"] = csrf

    def health(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for name, path in (("backend", "/health"), ("integrations", "/api/integrations/health")):
            started = time.perf_counter()
            try:
                response = self.client.get(path)
                elapsed = time.perf_counter() - started
                body: Any
                try:
                    body = response.json()
                except ValueError:
                    body = response.text[:1000]
                results[name] = {
                    "ok": response.is_success,
                    "status_code": response.status_code,
                    "latency_ms": round(elapsed * 1000, 3),
                    "body": body,
                }
            except httpx.HTTPError as exc:
                results[name] = {
                    "ok": False,
                    "status_code": None,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "error": str(exc),
                }
        return results

    def chat(self, prompt: str, iteration: int) -> Measurement:
        started = time.perf_counter()
        try:
            response = self.client.post("/api/chat", json={"message": prompt})
            elapsed = time.perf_counter() - started
            body = response.json() if response.content else {}
            success = response.is_success and not bool(body.get("fallback"))
            return Measurement(
                scenario="chat_sync",
                iteration=iteration,
                success=success,
                status_code=response.status_code,
                total_seconds=round(elapsed, 6),
                result_status="fallback" if body.get("fallback") else "completed",
                host_after=host_snapshot(),
                error=None if success else str(body)[:1000],
            )
        except (httpx.HTTPError, ValueError) as exc:
            return Measurement(
                scenario="chat_sync",
                iteration=iteration,
                success=False,
                status_code=None,
                total_seconds=round(time.perf_counter() - started, 6),
                host_after=host_snapshot(),
                error=str(exc),
            )

    def stream_task(
        self,
        *,
        scenario: str,
        prompt: str,
        iteration: int,
        chat_mode: str,
        execution_mode: str = "AUTOMATIC",
    ) -> Measurement:
        started = time.perf_counter()
        status_code: int | None = None
        try:
            start_response = self.client.post(
                "/api/tasks/start",
                json={
                    "request": prompt,
                    "chat_mode": chat_mode,
                    "execution_mode": execution_mode,
                    "attachments": [],
                },
            )
            status_code = start_response.status_code
            start_response.raise_for_status()
            task_id = str(start_response.json()["task_id"])
            first_token_at: float | None = None
            runtime_metrics: dict[str, Any] = {}
            terminal_type: str | None = None
            terminal_payload: dict[str, Any] = {}
            with self.client.stream(
                "GET",
                f"/api/tasks/{task_id}/events",
                headers={"Accept": "text/event-stream"},
                timeout=httpx.Timeout(self.timeout_seconds),
            ) as response:
                status_code = response.status_code
                response.raise_for_status()
                for event_name, envelope in _parse_sse(response):
                    kind = _event_type(event_name, envelope)
                    payload = _event_payload(envelope)
                    if kind == "model_token" and first_token_at is None:
                        first_token_at = time.perf_counter()
                    elif kind == "generation_completed":
                        metrics = payload.get("runtime_metrics")
                        if isinstance(metrics, dict):
                            runtime_metrics = metrics
                    if kind in TERMINAL_EVENTS:
                        terminal_type = kind
                        terminal_payload = payload
                        break
            elapsed = time.perf_counter() - started
            success = terminal_type == "task_completed"
            return Measurement(
                scenario=scenario,
                iteration=iteration,
                success=success,
                status_code=status_code,
                total_seconds=round(elapsed, 6),
                ttft_seconds=round(first_token_at - started, 6) if first_token_at else None,
                server_model_seconds=_as_float(runtime_metrics.get("total_duration_seconds")),
                tokens_per_second=_as_float(runtime_metrics.get("tokens_per_second")),
                token_count=_as_int(runtime_metrics.get("token_count")),
                warm_status=_as_str(runtime_metrics.get("warm_status")),
                result_status=terminal_type,
                host_after=host_snapshot(),
                error=None if success else json.dumps(terminal_payload, default=str)[:1000],
            )
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            return Measurement(
                scenario=scenario,
                iteration=iteration,
                success=False,
                status_code=status_code,
                total_seconds=round(time.perf_counter() - started, 6),
                host_after=host_snapshot(),
                error=str(exc),
            )

    def integration(
        self,
        *,
        scenario: str,
        payload: dict[str, Any],
        iteration: int,
    ) -> Measurement:
        started = time.perf_counter()
        try:
            response = self.client.post("/api/integrations/analyze", json=payload)
            elapsed = time.perf_counter() - started
            body = response.json() if response.content else {}
            success = response.is_success
            service_status = body.get("service_status") if isinstance(body, dict) else {}
            timings = body.get("timings_ms") if isinstance(body, dict) else {}
            return Measurement(
                scenario=scenario,
                iteration=iteration,
                success=success,
                status_code=response.status_code,
                total_seconds=round(elapsed, 6),
                released=body.get("released") if isinstance(body, dict) else None,
                result_status=_as_str(body.get("status")) if isinstance(body, dict) else None,
                service_status=service_status if isinstance(service_status, dict) else {},
                component_timings_ms={
                    str(k): float(v) for k, v in timings.items()
                    if isinstance(v, (int, float))
                } if isinstance(timings, dict) else {},
                host_after=host_snapshot(),
                error=None if success else json.dumps(body, default=str)[:1000],
            )
        except (httpx.HTTPError, ValueError) as exc:
            return Measurement(
                scenario=scenario,
                iteration=iteration,
                success=False,
                status_code=None,
                total_seconds=round(time.perf_counter() - started, 6),
                host_after=host_snapshot(),
                error=str(exc),
            )


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


def _as_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def process_diagnostic_payload(seed: int = 20) -> dict[str, Any]:
    rng = random.Random(seed)

    def row(shift: float = 0.0) -> list[float]:
        return [
            round(rng.gauss(0.0, 1.0) + shift, 6),
            round(rng.gauss(0.0, 1.0), 6),
            round(rng.gauss(0.0, 1.0), 6),
        ]

    normal_reference = [row() for _ in range(180)]
    signal_matrix = [row(4.0 if index >= 40 else 0.0) for index in range(100)]
    return {
        "domain": "process",
        "task": "fault_diagnosis",
        "inputs": {
            "normal_reference": normal_reference,
            "signal_matrix": signal_matrix,
            "channel_names": ["pressure", "flow", "level"],
            "sampling_rate_hz": 1.0,
            "maxlag": 1,
        },
        "policy_ref": "sovereign-system-benchmark-v1",
        "run_context": {"source": "scripts/benchmark_system.py", "synthetic": True},
    }


def graph_integration_payload() -> dict[str, Any]:
    return {
        "query": DEFAULT_INTEGRATION_QUERY,
        "include_graph_evidence": True,
        "diagnostic": None,
        "policy_profile": "internal_assistant",
        "consequential": False,
        "assurance_level": "standard",
    }


def full_integration_payload() -> dict[str, Any]:
    return {
        "query": DEFAULT_DIAGNOSTIC_QUERY,
        "include_graph_evidence": True,
        "diagnostic": process_diagnostic_payload(),
        "policy_profile": "internal_assistant",
        "consequential": True,
        "assurance_level": "thorough",
    }


def write_reports(
    output_dir: Path,
    *,
    metadata: dict[str, Any],
    health: dict[str, Any],
    measurements: list[Measurement],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_measurements(measurements)
    json_path = output_dir / "system_benchmark.json"
    csv_path = output_dir / "system_benchmark.csv"
    md_path = output_dir / "system_benchmark.md"

    document = {
        "metadata": metadata,
        "health": health,
        "summary": summary,
        "measurements": [asdict(item) for item in measurements],
    }
    json_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "scenario", "iteration", "success", "status_code", "total_seconds", "ttft_seconds",
        "server_model_seconds", "tokens_per_second", "token_count", "warm_status", "retries",
        "released", "result_status", "service_status", "component_timings_ms", "cpu_percent",
        "memory_used_mb", "memory_available_mb", "memory_percent", "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in measurements:
            writer.writerow({
                "scenario": item.scenario,
                "iteration": item.iteration,
                "success": item.success,
                "status_code": item.status_code,
                "total_seconds": item.total_seconds,
                "ttft_seconds": item.ttft_seconds,
                "server_model_seconds": item.server_model_seconds,
                "tokens_per_second": item.tokens_per_second,
                "token_count": item.token_count,
                "warm_status": item.warm_status,
                "retries": item.retries,
                "released": item.released,
                "result_status": item.result_status,
                "service_status": json.dumps(item.service_status, sort_keys=True),
                "component_timings_ms": json.dumps(item.component_timings_ms, sort_keys=True),
                "cpu_percent": item.host_after.cpu_percent,
                "memory_used_mb": item.host_after.memory_used_mb,
                "memory_available_mb": item.host_after.memory_available_mb,
                "memory_percent": item.host_after.memory_percent,
                "error": item.error,
            })

    lines = [
        "# SovereignAI system benchmark",
        "",
        f"- Generated: `{metadata['generated_at']}`",
        f"- Base URL: `{metadata['base_url']}`",
        f"- Runs per scenario: `{metadata['runs_per_scenario']}`",
        "",
        "## Health",
        "",
        "| Probe | OK | HTTP | Latency (ms) |",
        "|---|---:|---:|---:|",
    ]
    for name, result in health.items():
        lines.append(
            f"| {name} | {result.get('ok')} | {result.get('status_code')} | {result.get('latency_ms')} |"
        )
    lines.extend([
        "",
        "## Latency and reliability",
        "",
        "| Scenario | Success | p50 total (s) | p95 total (s) | p50 TTFT (s) | Avg tok/s |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for scenario, result in summary.items():
        lines.append(
            f"| {scenario} | {result['successes']}/{result['runs']} | "
            f"{_fmt(result['total_p50_s'])} | {_fmt(result['total_p95_s'])} | "
            f"{_fmt(result['ttft_p50_s'])} | {_fmt(result['tokens_per_second_mean'])} |"
        )

    component_rows = [
        item for item in measurements if item.success and item.component_timings_ms
    ]
    if component_rows:
        component_names = sorted({
            key for item in component_rows for key in item.component_timings_ms
        })
        lines.extend(["", "## Integrated component timing", ""])
        for scenario in sorted({item.scenario for item in component_rows}):
            lines.append(f"### {scenario}")
            lines.append("")
            lines.append("| Component | p50 (ms) | p95 (ms) |")
            lines.append("|---|---:|---:|")
            scenario_rows = [item for item in component_rows if item.scenario == scenario]
            for component in component_names:
                values = [
                    item.component_timings_ms[component]
                    for item in scenario_rows
                    if component in item.component_timings_ms
                ]
                if values:
                    lines.append(
                        f"| {component} | {_fmt(percentile(values, 0.50))} | "
                        f"{_fmt(percentile(values, 0.95))} |"
                    )
            lines.append("")

    failures = [item for item in measurements if not item.success]
    if failures:
        lines.extend(["", "## Failures / degraded runs", ""])
        for item in failures:
            lines.append(
                f"- `{item.scenario}` run {item.iteration}: HTTP={item.status_code}; "
                f"{item.error or item.result_status or 'unknown failure'}"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `chat_sync` measures the simple non-streaming chat API.",
        "- `task_general_stream` measures user-visible TTFT from task admission to the first `model_token` SSE event.",
        "- `task_authorized_stream` exercises the authorized/RAG path and records TTFT when model generation occurs.",
        "- `integration_graph_controlplane` exercises ControlPlane precheck → GraphRAG → ControlPlane release.",
        "- `integration_full_industrial` additionally sends deterministic synthetic process data through the Time-Series Diagnostic Agent.",
        "- The harness does not stop services automatically. Run it again while intentionally stopping one service to capture degraded/fail-closed behavior.",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the local SovereignAI stack end to end.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=3, help="Runs per scenario.")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/system_benchmark"))
    parser.add_argument("--bearer-token", default=None)
    parser.add_argument("--cookie", default=None, help="Raw Cookie header for local-auth mode.")
    parser.add_argument("--csrf-token", default=None, help="CSRF token when supplying a raw local-auth cookie.")
    parser.add_argument("--email", default=None, help="Optional local-auth email; use together with --password.")
    parser.add_argument("--password", default=None, help="Optional local-auth password; use together with --email.")
    parser.add_argument("--csrf-cookie-name", default="sovereign_csrf")
    parser.add_argument("--skip-chat", action="store_true")
    parser.add_argument("--skip-stream", action="store_true")
    parser.add_argument("--skip-integrations", action="store_true")
    parser.add_argument(
        "--include-diagnostic",
        action="store_true",
        help="Also benchmark GraphRAG + Time-Series Diagnostic Agent + ControlPlane.",
    )
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="Exit 0 even if health probes or benchmark scenarios fail.",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if bool(args.email) != bool(args.password):
        parser.error("--email and --password must be supplied together")
    return args


def main() -> int:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    initial_host = host_snapshot()
    runner = BenchmarkRunner(
        base_url=args.base_url,
        timeout_seconds=args.timeout,
        bearer_token=args.bearer_token,
        cookie=args.cookie,
        csrf_token=args.csrf_token,
    )
    measurements: list[Measurement] = []
    try:
        if args.email and args.password:
            runner.login(args.email, args.password, args.csrf_cookie_name)
        health = runner.health()
        if not args.skip_chat:
            for iteration in range(1, args.runs + 1):
                measurements.append(runner.chat(DEFAULT_CHAT_PROMPT, iteration))
        if not args.skip_stream:
            for iteration in range(1, args.runs + 1):
                measurements.append(runner.stream_task(
                    scenario="task_general_stream",
                    prompt=DEFAULT_CHAT_PROMPT,
                    iteration=iteration,
                    chat_mode="GENERAL",
                    execution_mode="STANDARD",
                ))
            for iteration in range(1, args.runs + 1):
                measurements.append(runner.stream_task(
                    scenario="task_authorized_stream",
                    prompt=DEFAULT_RAG_PROMPT,
                    iteration=iteration,
                    chat_mode="AUTHORIZED",
                    execution_mode="STANDARD",
                ))
        if not args.skip_integrations:
            for iteration in range(1, args.runs + 1):
                measurements.append(runner.integration(
                    scenario="integration_graph_controlplane",
                    payload=graph_integration_payload(),
                    iteration=iteration,
                ))
            if args.include_diagnostic:
                for iteration in range(1, args.runs + 1):
                    measurements.append(runner.integration(
                        scenario="integration_full_industrial",
                        payload=full_integration_payload(),
                        iteration=iteration,
                    ))
    finally:
        runner.close()

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at.isoformat(),
        "base_url": args.base_url,
        "runs_per_scenario": args.runs,
        "include_diagnostic": args.include_diagnostic,
        "host_before": asdict(initial_host),
        "python": sys.version.split()[0],
    }
    paths = write_reports(
        args.output_dir,
        metadata=metadata,
        health=health,
        measurements=measurements,
    )
    summary = summarize_measurements(measurements)
    print(json.dumps({"health": health, "summary": summary, "reports": {k: str(v) for k, v in paths.items()}}, indent=2))

    health_ok = all(bool(item.get("ok")) for item in health.values())
    measurements_ok = all(item.success for item in measurements) if measurements else True
    if args.allow_degraded:
        return 0
    return 0 if health_ok and measurements_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
