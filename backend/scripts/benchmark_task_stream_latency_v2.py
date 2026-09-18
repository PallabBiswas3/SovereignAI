from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

import httpx

from benchmark_task_stream_latency import DEFAULT_PROMPT, _run_task, _summarize

DEFAULT_MODEL = "qwen3:4b-instruct"
EXACT_RUNNER_OPTIONS: dict[str, int | float] = {
    "num_ctx": 2048,
    "num_predict": 1,
    "temperature": 0.2,
    "num_thread": 5,
    "num_batch": 128,
}


def _require_loopback(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"Benchmark endpoint must be loopback-local: {url}")


def _resident_names(payload: dict[str, Any]) -> set[str]:
    models = payload.get("models") if isinstance(payload.get("models"), list) else []
    names: set[str] = set()
    for item in models:
        if not isinstance(item, dict):
            continue
        value = str(item.get("name") or item.get("model") or "").strip()
        if value:
            names.add(value)
    return names


def _matches_model(names: set[str], model: str) -> bool:
    wanted = {model, f"{model}:latest"} if ":" not in model else {model}
    return bool(names & wanted)


async def _exact_runner_prewarm(
    client: httpx.AsyncClient,
    *,
    ollama_url: str,
    model: str,
    keep_alive: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = perf_counter()
    response = await client.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": "/no_think\nReply with one token: READY",
            "system": "You are a local industrial assistant. Be concise.",
            "stream": False,
            "think": False,
            "keep_alive": keep_alive,
            "options": dict(EXACT_RUNNER_OPTIONS),
        },
        timeout=timeout_seconds,
    )
    wall = perf_counter() - started
    if not response.is_success:
        raise RuntimeError(
            f"Direct Ollama prewarm failed with HTTP {response.status_code}: {response.text[:2000]}"
        )
    data = response.json()
    ps_response = await client.get(f"{ollama_url.rstrip('/')}/api/ps", timeout=10.0)
    ps_response.raise_for_status()
    residents = _resident_names(ps_response.json())
    if not _matches_model(residents, model):
        raise RuntimeError(f"Direct prewarm completed but {model} is not resident according to /api/ps")
    return {
        "wall_seconds": round(wall, 6),
        "load_duration_seconds": round(float(data.get("load_duration", 0) or 0) / 1_000_000_000, 6),
        "response": str(data.get("response") or "").strip(),
        "model": model,
        "keep_alive": keep_alive,
        "options": EXACT_RUNNER_OPTIONS,
        "resident_models": sorted(residents),
    }


async def _resource_snapshot(client: httpx.AsyncClient, api_url: str) -> dict[str, Any]:
    response = await client.get(f"{api_url.rstrip('/')}/api/models/status", timeout=30.0)
    if not response.is_success:
        raise RuntimeError(
            f"GET /api/models/status returned HTTP {response.status_code}: {response.text[:2000]}"
        )
    payload = response.json()
    resources = payload.get("resources") if isinstance(payload.get("resources"), dict) else {}
    models = payload.get("models") if isinstance(payload.get("models"), list) else []
    return {"resources": resources, "models": models}


async def _wait_for_resident_headroom(
    client: httpx.AsyncClient,
    *,
    api_url: str,
    model: str,
    headroom_mb: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = perf_counter()
    best_available = 0.0
    latest: dict[str, Any] = {}
    while perf_counter() - started <= timeout_seconds:
        latest = await _resource_snapshot(client, api_url)
        resources = latest.get("resources") if isinstance(latest.get("resources"), dict) else {}
        available = float(resources.get("ram_available_mb") or 0.0)
        reserve = float(resources.get("ram_reserve_mb") or 0.0)
        best_available = max(best_available, available)
        model_rows = latest.get("models") if isinstance(latest.get("models"), list) else []
        model_row = next(
            (
                item for item in model_rows
                if isinstance(item, dict) and str(item.get("model_tag") or "") == model
            ),
            None,
        )
        resident = bool(
            isinstance(model_row, dict)
            and (
                str(model_row.get("warm_status") or "").lower() == "warm"
                or str(model_row.get("lifecycle_state") or "").upper() in {"WARM", "LOADED", "READY"}
                and model_row.get("memory_usage_mb") is not None
            )
        )
        required = reserve + max(0.0, headroom_mb)
        if resident and available >= required:
            return {
                "ready": True,
                "resident": True,
                "available_mb": round(available, 2),
                "reserve_mb": round(reserve, 2),
                "benchmark_headroom_mb": round(max(0.0, headroom_mb), 2),
                "required_available_mb": round(required, 2),
                "wait_seconds": round(perf_counter() - started, 3),
                "model_status": model_row,
            }
        await asyncio.sleep(0.25)

    resources = latest.get("resources") if isinstance(latest.get("resources"), dict) else {}
    available = float(resources.get("ram_available_mb") or 0.0)
    reserve = float(resources.get("ram_reserve_mb") or 0.0)
    return {
        "ready": False,
        "available_mb": round(available, 2),
        "reserve_mb": round(reserve, 2),
        "benchmark_headroom_mb": round(max(0.0, headroom_mb), 2),
        "required_available_mb": round(reserve + max(0.0, headroom_mb), 2),
        "best_available_mb": round(best_available, 2),
        "wait_seconds": round(perf_counter() - started, 3),
    }


def _completed_state(run: dict[str, Any]) -> str | None:
    events = run.get("events") if isinstance(run.get("events"), list) else []
    for event in reversed(events):
        if not isinstance(event, dict) or event.get("type") != "task_completed":
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        value = result.get("status")
        return str(value) if value is not None else None
    return None


def _validate_run(run: dict[str, Any]) -> None:
    terminal = str(run.get("terminal_event_type") or "")
    state = (_completed_state(run) or "").lower()
    tokens = int(run.get("model_token_event_count") or 0)
    metrics = run.get("agent_runtime_metrics") if isinstance(run.get("agent_runtime_metrics"), dict) else {}
    if terminal != "task_completed" or state != "completed" or tokens <= 0 or not metrics:
        warning = None
        events = run.get("events") if isinstance(run.get("events"), list) else []
        for event in reversed(events):
            if isinstance(event, dict) and event.get("type") == "warning":
                payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                warning = payload.get("summary")
                break
        raise RuntimeError(
            "Phase 12 run was not a valid generated-response measurement: "
            f"terminal={terminal or 'missing'}, state={state or 'missing'}, model_token_events={tokens}. "
            f"{warning or ''}".strip()
        )


async def _main(args: argparse.Namespace) -> dict[str, Any]:
    _require_loopback(args.api_url)
    _require_loopback(args.ollama_url)
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        health = await client.get(f"{args.api_url.rstrip('/')}/health", timeout=5.0)
        if not health.is_success:
            raise RuntimeError(f"SovereignAI backend is not reachable at {args.api_url}")
        tags = await client.get(f"{args.ollama_url.rstrip('/')}/api/tags", timeout=5.0)
        tags.raise_for_status()

        prewarm = await _exact_runner_prewarm(
            client,
            ollama_url=args.ollama_url,
            model=args.model,
            keep_alive=args.keep_alive,
            timeout_seconds=args.timeout,
        )
        gate = await _wait_for_resident_headroom(
            client,
            api_url=args.api_url,
            model=args.model,
            headroom_mb=args.ram_headroom_mb,
            timeout_seconds=args.ram_wait_seconds,
        )
        if not gate["ready"]:
            raise RuntimeError(
                "Exact-runner prewarm succeeded, but the resident model does not leave enough RAM for the "
                f"unchanged SovereignAI reserve plus benchmark headroom: available={gate['available_mb']:.0f} MB, "
                f"required={gate['required_available_mb']:.0f} MB. Close memory-heavy applications and retry; "
                "do not weaken the production reserve for this benchmark."
            )

        warmups: list[dict[str, Any]] = []
        for index in range(max(0, args.warmup_runs)):
            run = await _run_task(
                client,
                api_url=args.api_url,
                prompt=f"{args.prompt}\nFull-path warm-up {index + 1}.",
                timeout_seconds=args.timeout,
                measured=False,
            )
            _validate_run(run)
            warmups.append(run)

        runs: list[dict[str, Any]] = []
        gates: list[dict[str, Any]] = []
        for index in range(max(1, args.repeats)):
            round_gate = await _wait_for_resident_headroom(
                client,
                api_url=args.api_url,
                model=args.model,
                headroom_mb=args.ram_headroom_mb,
                timeout_seconds=args.ram_wait_seconds,
            )
            gates.append(round_gate)
            if not round_gate["ready"]:
                raise RuntimeError(
                    f"RAM/residency gate failed before measured round {index + 1}: "
                    f"available={round_gate['available_mb']:.0f} MB, "
                    f"required={round_gate['required_available_mb']:.0f} MB."
                )
            run = await _run_task(
                client,
                api_url=args.api_url,
                prompt=f"{args.prompt}\nBenchmark run {index + 1}.",
                timeout_seconds=args.timeout,
                measured=True,
            )
            _validate_run(run)
            runs.append(run)
            progress = {
                "benchmark_design": "phase12-real-task-sse-latency-v2-resident-gated",
                "status": "running" if index + 1 < max(1, args.repeats) else "complete",
                "model": args.model,
                "prewarm": prewarm,
                "initial_ram_gate": gate,
                "warmup_runs": warmups,
                "round_ram_gates": gates,
                "runs": runs,
                "summary": _summarize(runs),
                "interpretation_note": (
                    "The model is exact-runner prewarmed directly through loopback Ollama outside measurement. "
                    "Production admission policy is unchanged. Every measured round is accepted only when the model "
                    "is resident, RAM exceeds the production reserve plus benchmark headroom, task state is completed, "
                    "at least one model_token SSE frame arrived, and provider runtime metrics are present."
                ),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(progress, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "benchmark_design": "phase12-real-task-sse-latency-v2-resident-gated",
        "status": "complete",
        "model": args.model,
        "requested_repeats": max(1, args.repeats),
        "completed_repeats": len(runs),
        "prewarm": prewarm,
        "initial_ram_gate": gate,
        "warmup_runs": warmups,
        "round_ram_gates": gates,
        "runs": runs,
        "summary": _summarize(runs),
        "interpretation_note": (
            "The model is exact-runner prewarmed directly through loopback Ollama outside measurement. "
            "Production admission policy is unchanged. Every measured round is accepted only when the model "
            "is resident, RAM exceeds the production reserve plus benchmark headroom, task state is completed, "
            "at least one model_token SSE frame arrived, and provider runtime metrics are present."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 12 v2: resident-gated real task -> AgentExecutor -> SSE latency benchmark."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--keep-alive", default="15m")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--ram-headroom-mb", type=float, default=96.0)
    parser.add_argument("--ram-wait-seconds", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=Path("workspace/phase12_task_stream_latency_v2.json"))
    args = parser.parse_args()

    report = asyncio.run(_main(args))
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
