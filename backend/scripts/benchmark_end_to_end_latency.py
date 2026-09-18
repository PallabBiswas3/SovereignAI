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

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

SYSTEM_PROMPT = "You are SovereignAI, a local enterprise assistant. Be concise and never invent sources."
DEFAULT_PROMPT = "Explain in four concise technical points why an outer-race bearing fault creates periodic vibration impulses."
DEFAULT_MODEL = "qwen3:4b-instruct"


def _require_local(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"Benchmark endpoints must be loopback-local: {url}")


def _median(values: list[float]) -> float | None:
    return round(median(values), 6) if values else None


def _response_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        return response.text.strip()[:2000]
    return json.dumps(payload, ensure_ascii=False)


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
    """Warm the target model before chat calibration so Phase 11 measures the warm path by design."""
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
    for name in ("wall_seconds", "time_to_first_token_seconds", "ollama_total_duration_seconds", "load_duration_seconds", "tokens_per_second"):
        values = [float(run[name]) for run in runs if isinstance(run.get(name), (int, float))]
        summary[f"median_{name}"] = _median(values)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired warm-path SovereignAI /api/chat versus direct Ollama latency benchmark.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to pre-warm before /api/chat calibration.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--keep-alive", default="60s")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=Path("workspace/e2e_latency.json"))
    args = parser.parse_args()

    _require_local(args.api_url)
    _require_local(args.ollama_url)
    repeats = max(2, args.repeats)

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

        # Warm the direct path once with exactly the runtime settings emitted by /api/chat.
        _direct_ollama_stream(
            client,
            args.ollama_url,
            prompt=args.prompt,
            effective_model=direct_settings["effective_model"],
            system=SYSTEM_PROMPT,
            options=direct_settings["options"],
            keep_alive=args.keep_alive,
            timeout_seconds=args.timeout,
        )

        rounds: list[dict[str, Any]] = []
        chat_runs: list[dict[str, Any]] = []
        direct_runs: list[dict[str, Any]] = []
        paired_wall_ratios: list[float] = []
        for index in range(repeats):
            order = ["sovereignai", "direct_ollama"] if index % 2 == 0 else ["direct_ollama", "sovereignai"]
            pair: dict[str, Any] = {}
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
                        raise RuntimeError(f"Round {index + 1} /api/chat fell back instead of using the local model")
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
            rounds.append({
                "round": index + 1,
                "order": order,
                "sovereignai": pair["sovereignai"],
                "direct_ollama": pair["direct_ollama"],
                "sovereignai_vs_direct_wall_ratio": round(ratio, 4) if ratio else None,
            })

    chat_summary = _summarize_chat(chat_runs)
    direct_summary = _summarize_direct(direct_runs)
    endpoint_total = chat_summary.get("median_endpoint_total_seconds")
    app_overhead = chat_summary.get("median_application_overhead_excluding_model_seconds")
    overhead_fraction = (
        float(app_overhead) / float(endpoint_total)
        if isinstance(app_overhead, (int, float)) and isinstance(endpoint_total, (int, float)) and endpoint_total > 0
        else None
    )

    report = {
        "benchmark_design": "paired-e2e-warm-path-v2",
        "prompt": args.prompt,
        "api_url": args.api_url,
        "ollama_url": args.ollama_url,
        "repeats": repeats,
        "prewarm": prewarm,
        "calibration": {
            "model": calibration["model"],
            "provider": calibration["provider"],
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
            "This is a warm-path paired benchmark. The target model is explicitly pre-warmed before /api/chat "
            "calibration so the backend RAM admission controller evaluates it as resident. Direct Ollama then uses "
            "the effective model, context, output budget, temperature and CPU tuning emitted by calibration."
        ),
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())