from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import urlparse

import httpx
import psutil

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

SYSTEM_PROMPT = "You are SovereignAI, a local enterprise assistant. Be concise and never invent sources."
DEFAULT_PROMPT = (
    "Answer with exactly four short bullets, one sentence each: why does an outer-race "
    "bearing fault create periodic vibration impulses?"
)
DEFAULT_MODEL = "qwen3:4b-instruct"


def _require_local(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"Benchmark endpoints must be loopback-local: {url}")


def _median(values: list[float]) -> float | None:
    return round(median(values), 6) if values else None


def _memory_snapshot() -> dict[str, float]:
    memory = psutil.virtual_memory()
    return {
        "available_mb": round(memory.available / (1024 * 1024), 2),
        "used_mb": round(memory.used / (1024 * 1024), 2),
        "percent": round(float(memory.percent), 2),
    }


def _response_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        return response.text.strip()[:2000]
    return json.dumps(payload, ensure_ascii=False)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _chat_call(
    client: httpx.Client,
    api_url: str,
    *,
    prompt: str,
    conversation_id: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"message": prompt}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    started = time.perf_counter()
    response = client.post(f"{api_url.rstrip('/')}/api/chat", json=payload, timeout=timeout_seconds)
    wall = time.perf_counter() - started
    if response.status_code in {401, 403}:
        raise RuntimeError(
            "The local /api/chat endpoint requires authentication. Run this benchmark with the development "
            "auth-disabled configuration or extend the script with your local session cookie."
        )
    if not response.is_success:
        raise RuntimeError(
            f"SovereignAI /api/chat returned HTTP {response.status_code}: {_response_error(response)}"
        )
    data = response.json()
    runtime = data.get("runtime_metrics") if isinstance(data.get("runtime_metrics"), dict) else {}
    return {
        "wall_seconds": round(wall, 6),
        "conversation_id": data.get("conversation_id"),
        "response": data.get("response"),
        "model": data.get("model"),
        "provider": data.get("provider"),
        "fallback": bool(data.get("fallback")),
        "runtime_metrics": runtime,
    }


def _prewarm_model(
    client: httpx.Client,
    ollama_url: str,
    *,
    model: str,
    keep_alive: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Load the target model outside measured requests so Phase 11 measures only the warm path."""
    payload = {
        "model": model,
        "prompt": "/no_think\nReply with exactly READY.",
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "options": {
            "num_ctx": 2048,
            "num_predict": 8,
            "temperature": 0.0,
        },
    }
    started = time.perf_counter()
    response = client.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json=payload,
        timeout=timeout_seconds,
    )
    wall = time.perf_counter() - started
    if not response.is_success:
        raise RuntimeError(
            f"Ollama prewarm failed with HTTP {response.status_code}: {_response_error(response)}"
        )
    data = response.json()
    return {
        "model": model,
        "wall_seconds": round(wall, 6),
        "load_duration_seconds": round(float(data.get("load_duration", 0) or 0) / 1_000_000_000, 6),
        "response": str(data.get("response") or "").strip(),
        "memory_after": _memory_snapshot(),
    }


def _resident_models(client: httpx.Client, ollama_url: str) -> set[str]:
    response = client.get(f"{ollama_url.rstrip('/')}/api/ps", timeout=5.0)
    response.raise_for_status()
    items = response.json().get("models", [])
    return {
        str(item.get("name") or item.get("model") or "").strip()
        for item in items
        if isinstance(item, dict)
    }


def _model_is_resident(client: httpx.Client, ollama_url: str, model: str) -> bool:
    names = _resident_models(client, ollama_url)
    return model in names or (":" not in model and f"{model}:latest" in names)


def _unload_model(
    client: httpx.Client,
    ollama_url: str,
    *,
    model: str,
    timeout_seconds: float,
) -> None:
    response = client.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={"model": model, "keep_alive": 0, "stream": False},
        timeout=timeout_seconds,
    )
    if not response.is_success:
        raise RuntimeError(
            f"Ollama unload failed with HTTP {response.status_code}: {_response_error(response)}"
        )


def _reset_and_rewarm(
    client: httpx.Client,
    ollama_url: str,
    *,
    model: str,
    keep_alive: str,
    timeout_seconds: float,
    settle_seconds: float,
) -> dict[str, Any]:
    """Clear per-run model/prompt state, then reload before measurements start."""
    before = _memory_snapshot()
    _unload_model(client, ollama_url, model=model, timeout_seconds=timeout_seconds)

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if not _model_is_resident(client, ollama_url, model):
            break
        time.sleep(0.25)
    if _model_is_resident(client, ollama_url, model):
        raise RuntimeError(f"Model {model} remained resident after unload request")

    if settle_seconds > 0:
        time.sleep(settle_seconds)
    after_unload = _memory_snapshot()
    warm = _prewarm_model(
        client,
        ollama_url,
        model=model,
        keep_alive=keep_alive,
        timeout_seconds=timeout_seconds,
    )
    if not _model_is_resident(client, ollama_url, model):
        raise RuntimeError(f"Model {model} was not resident after prewarm")
    return {
        "memory_before_reset": before,
        "memory_after_unload": after_unload,
        "prewarm": warm,
        "memory_after_rewarm": _memory_snapshot(),
    }


def _direct_ollama_stream(
    client: httpx.Client,
    ollama_url: str,
    *,
    prompt: str,
    effective_model: str,
    system: str,
    options: dict[str, int | float],
    keep_alive: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": effective_model,
        "prompt": prompt,
        "system": system,
        "stream": True,
        "think": False,
        "keep_alive": keep_alive,
        "options": options,
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    final_item: dict[str, Any] = {}
    pieces: list[str] = []
    with client.stream(
        "POST",
        f"{ollama_url.rstrip('/')}/api/generate",
        json=payload,
        timeout=timeout_seconds,
    ) as response:
        if not response.is_success:
            raise RuntimeError(
                f"Direct Ollama returned HTTP {response.status_code}: {_response_error(response)}"
            )
        for line in response.iter_lines():
            if not line.strip():
                continue
            item = json.loads(line)
            token = str(item.get("response") or "")
            if token:
                first_token_at = first_token_at or time.perf_counter()
                pieces.append(token)
            if item.get("done"):
                final_item = item
                break
    wall = time.perf_counter() - started
    eval_seconds = float(final_item.get("eval_duration", 0) or 0) / 1_000_000_000
    eval_count = int(final_item.get("eval_count", 0) or 0)
    total_seconds = float(final_item.get("total_duration", 0) or 0) / 1_000_000_000
    load_seconds = float(final_item.get("load_duration", 0) or 0) / 1_000_000_000
    return {
        "wall_seconds": round(wall, 6),
        "time_to_first_token_seconds": round(first_token_at - started, 6) if first_token_at else None,
        "ollama_total_duration_seconds": round(total_seconds, 6) if total_seconds else None,
        "load_duration_seconds": round(load_seconds, 6) if load_seconds else 0.0,
        "tokens_per_second": round(eval_count / eval_seconds, 3) if eval_seconds and eval_count else None,
        "token_count": eval_count,
        "prompt_token_count": int(final_item.get("prompt_eval_count", 0) or 0),
        "done_reason": final_item.get("done_reason"),
        "response": "".join(pieces).strip(),
    }


def _direct_settings_from_chat(chat: dict[str, Any]) -> dict[str, Any]:
    runtime = chat.get("runtime_metrics") or {}
    provider = runtime.get("provider_generation") or {}
    runtime_profile = provider.get("runtime_profile") or {}
    context_plan = provider.get("context_plan") or {}
    cpu_tuning = provider.get("cpu_tuning") or {}
    effective_model = str(provider.get("effective_model") or chat.get("model") or "")
    if not effective_model:
        raise RuntimeError("Calibration chat response did not expose an effective model")

    num_ctx = int(context_plan.get("selected_num_ctx") or runtime_profile.get("num_ctx") or 2048)
    num_predict = int(runtime_profile.get("num_predict") or 384)
    temperature = float(runtime_profile.get("temperature") or 0.2)
    options: dict[str, int | float] = {
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "temperature": temperature,
    }
    if cpu_tuning.get("num_thread"):
        options["num_thread"] = int(cpu_tuning["num_thread"])
    if cpu_tuning.get("num_batch"):
        options["num_batch"] = int(cpu_tuning["num_batch"])
    return {
        "effective_model": effective_model,
        "options": options,
        "context_plan": context_plan,
        "runtime_profile": runtime_profile,
        "cpu_tuning": cpu_tuning,
    }


def _summarize_chat(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "endpoint_total_seconds",
        "request_setup_seconds",
        "routing_seconds",
        "provider_setup_seconds",
        "generation_seconds",
        "persistence_commit_seconds",
        "application_overhead_excluding_model_seconds",
    ]
    summary: dict[str, Any] = {}
    for name in metric_names:
        values = [
            float(run["runtime_metrics"][name])
            for run in runs
            if isinstance(run.get("runtime_metrics"), dict)
            and isinstance(run["runtime_metrics"].get(name), (int, float))
        ]
        summary[f"median_{name}"] = _median(values)

    walls = [float(run["wall_seconds"]) for run in runs]
    summary["median_client_wall_seconds"] = _median(walls)
    provider_stats = [run.get("runtime_metrics", {}).get("provider_generation", {}) for run in runs]
    for name in ("time_to_first_token_seconds", "queue_wait_seconds", "load_duration_seconds", "tokens_per_second"):
        values = [float(stats[name]) for stats in provider_stats if isinstance(stats.get(name), (int, float))]
        summary[f"median_provider_{name}"] = _median(values)
    return summary


def _summarize_direct(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in (
        "wall_seconds",
        "time_to_first_token_seconds",
        "ollama_total_duration_seconds",
        "load_duration_seconds",
        "tokens_per_second",
    ):
        values = [float(run[name]) for run in runs if isinstance(run.get(name), (int, float))]
        summary[f"median_{name}"] = _median(values)
    return summary


def _build_report(
    *,
    args: argparse.Namespace,
    prewarm: dict[str, Any],
    calibration: dict[str, Any],
    direct_settings: dict[str, Any],
    rounds: list[dict[str, Any]],
    chat_runs: list[dict[str, Any]],
    direct_runs: list[dict[str, Any]],
    paired_wall_ratios: list[float],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    chat_summary = _summarize_chat(chat_runs)
    direct_summary = _summarize_direct(direct_runs)
    endpoint_total = chat_summary.get("median_endpoint_total_seconds")
    app_overhead = chat_summary.get("median_application_overhead_excluding_model_seconds")
    overhead_fraction = (
        float(app_overhead) / float(endpoint_total)
        if isinstance(app_overhead, (int, float))
        and isinstance(endpoint_total, (int, float))
        and endpoint_total > 0
        else None
    )
    return {
        "benchmark_design": "paired-e2e-warm-path-v3-reset-between-rounds",
        "status": status,
        "error": error,
        "prompt": args.prompt,
        "api_url": args.api_url,
        "ollama_url": args.ollama_url,
        "requested_repeats": max(2, args.repeats),
        "completed_rounds": len(rounds),
        "reset_between_rounds": True,
        "settle_seconds": args.settle_seconds,
        "initial_prewarm": prewarm,
        "calibration": {
            "model": calibration.get("model"),
            "provider": calibration.get("provider"),
            "direct_runtime_settings": direct_settings,
        },
        "rounds": rounds,
        "summary": {
            "sovereignai": chat_summary,
            "direct_ollama": direct_summary,
            "median_sovereignai_vs_direct_wall_ratio": _median(paired_wall_ratios),
            "median_application_overhead_fraction": round(overhead_fraction, 4) if overhead_fraction is not None else None,
        },
        "interpretation_note": (
            "This is a warm-path paired benchmark. The model is unloaded and re-warmed outside measured calls "
            "before every pair so prompt-cache growth from prior rounds does not consume the OS RAM reserve. "
            "Direct Ollama uses the effective model, context, output budget, temperature and CPU tuning emitted "
            "by the calibration /api/chat request. Reset/prewarm time is recorded but excluded from target latency."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired warm-path SovereignAI /api/chat versus direct Ollama latency benchmark."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to pre-warm before /api/chat calibration.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--keep-alive", default="60s")
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=Path("workspace/e2e_latency.json"))
    args = parser.parse_args()

    _require_local(args.api_url)
    _require_local(args.ollama_url)
    repeats = max(2, args.repeats)

    rounds: list[dict[str, Any]] = []
    chat_runs: list[dict[str, Any]] = []
    direct_runs: list[dict[str, Any]] = []
    paired_wall_ratios: list[float] = []

    with httpx.Client(follow_redirects=False) as client:
        api_health = client.get(f"{args.api_url.rstrip('/')}/docs", timeout=5.0)
        if not api_health.is_success:
            raise RuntimeError(f"SovereignAI backend is not reachable at {args.api_url}")
        ollama_health = client.get(f"{args.ollama_url.rstrip('/')}/api/tags", timeout=5.0)
        ollama_health.raise_for_status()

        prewarm = _prewarm_model(
            client,
            args.ollama_url,
            model=args.model,
            keep_alive=args.keep_alive,
            timeout_seconds=args.timeout,
        )

        calibration = _chat_call(
            client,
            args.api_url,
            prompt=args.prompt,
            conversation_id=None,
            timeout_seconds=args.timeout,
        )
        if calibration["fallback"]:
            raise RuntimeError("Calibration /api/chat request fell back instead of using the local model")
        direct_settings = _direct_settings_from_chat(calibration)
        if direct_settings["effective_model"].lower() != args.model.lower():
            raise RuntimeError(
                "The routed /api/chat model differs from the model pre-warmed by this benchmark: "
                f"prewarmed={args.model}, routed={direct_settings['effective_model']}. "
                "Pass --model with the routed model or use a prompt that routes to GENERAL."
            )
        conversation_id = str(calibration["conversation_id"])

        for index in range(repeats):
            reset = _reset_and_rewarm(
                client,
                args.ollama_url,
                model=direct_settings["effective_model"],
                keep_alive=args.keep_alive,
                timeout_seconds=args.timeout,
                settle_seconds=max(0.0, args.settle_seconds),
            )
            order = ["sovereignai", "direct_ollama"] if index % 2 == 0 else ["direct_ollama", "sovereignai"]
            pair: dict[str, Any] = {
                "round": index + 1,
                "order": order,
                "reset": reset,
                "memory_before_pair": _memory_snapshot(),
            }
            try:
                for target in order:
                    if target == "sovereignai":
                        run = _chat_call(
                            client,
                            args.api_url,
                            prompt=args.prompt,
                            conversation_id=conversation_id,
                            timeout_seconds=args.timeout,
                        )
                        if run["fallback"]:
                            raise RuntimeError(
                                f"Round {index + 1} /api/chat fell back instead of using the local model"
                            )
                        chat_runs.append(run)
                        pair[target] = run
                    else:
                        run = _direct_ollama_stream(
                            client,
                            args.ollama_url,
                            prompt=args.prompt,
                            effective_model=direct_settings["effective_model"],
                            system=SYSTEM_PROMPT,
                            options=direct_settings["options"],
                            keep_alive=args.keep_alive,
                            timeout_seconds=args.timeout,
                        )
                        direct_runs.append(run)
                        pair[target] = run

                sovereign_wall = float(pair["sovereignai"]["wall_seconds"])
                direct_wall = float(pair["direct_ollama"]["wall_seconds"])
                ratio = sovereign_wall / direct_wall if direct_wall > 0 else 0.0
                if ratio > 0:
                    paired_wall_ratios.append(ratio)
                pair["sovereignai_vs_direct_wall_ratio"] = round(ratio, 4) if ratio else None
                pair["memory_after_pair"] = _memory_snapshot()
                rounds.append(pair)

                progress = _build_report(
                    args=args,
                    prewarm=prewarm,
                    calibration=calibration,
                    direct_settings=direct_settings,
                    rounds=rounds,
                    chat_runs=chat_runs,
                    direct_runs=direct_runs,
                    paired_wall_ratios=paired_wall_ratios,
                    status="running" if len(rounds) < repeats else "complete",
                )
                _write_report(args.output, progress)
            except Exception as exc:
                pair["memory_at_failure"] = _memory_snapshot()
                pair["error"] = str(exc)
                partial_rounds = rounds + [pair]
                failure = _build_report(
                    args=args,
                    prewarm=prewarm,
                    calibration=calibration,
                    direct_settings=direct_settings,
                    rounds=partial_rounds,
                    chat_runs=chat_runs,
                    direct_runs=direct_runs,
                    paired_wall_ratios=paired_wall_ratios,
                    status="incomplete",
                    error=str(exc),
                )
                _write_report(args.output, failure)
                print(json.dumps(failure, indent=2, ensure_ascii=False))
                return 2

    report = _build_report(
        args=args,
        prewarm=prewarm,
        calibration=calibration,
        direct_settings=direct_settings,
        rounds=rounds,
        chat_runs=chat_runs,
        direct_runs=direct_runs,
        paired_wall_ratios=paired_wall_ratios,
        status="complete",
    )
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    _write_report(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
