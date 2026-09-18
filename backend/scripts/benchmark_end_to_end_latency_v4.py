from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx

from benchmark_end_to_end_latency import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    SYSTEM_PROMPT,
    _chat_call,
    _direct_ollama_stream,
    _direct_settings_from_chat,
    _memory_snapshot,
    _median,
    _prewarm_model,
    _require_local,
    _reset_and_rewarm,
    _summarize_chat,
    _summarize_direct,
    _write_report,
)


def _measured_load_seconds(target: str, run: dict[str, Any]) -> float | None:
    if target == "direct_ollama":
        value = run.get("load_duration_seconds")
    else:
        runtime = run.get("runtime_metrics") if isinstance(run.get("runtime_metrics"), dict) else {}
        provider = runtime.get("provider_generation") if isinstance(runtime.get("provider_generation"), dict) else {}
        value = provider.get("load_duration_seconds")
    return float(value) if isinstance(value, (int, float)) else None


def _build_report(
    *,
    args: argparse.Namespace,
    calibration: dict[str, Any],
    direct_settings: dict[str, Any],
    rounds: list[dict[str, Any]],
    chat_runs: list[dict[str, Any]],
    direct_runs: list[dict[str, Any]],
    paired_ratios: list[float],
    invalid_rounds: list[int],
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
        "benchmark_design": "paired-e2e-warm-path-v4-exact-runner-prewarm",
        "status": status,
        "error": error,
        "prompt": args.prompt,
        "requested_repeats": max(2, args.repeats),
        "completed_rounds": len(rounds),
        "invalid_warm_path_rounds": invalid_rounds,
        "max_measured_load_seconds": args.max_measured_load_seconds,
        "calibration": {
            "model": calibration.get("model"),
            "provider": calibration.get("provider"),
            "direct_runtime_settings": direct_settings,
        },
        "rounds": rounds,
        "summary": {
            "sovereignai": chat_summary,
            "direct_ollama": direct_summary,
            "median_sovereignai_vs_direct_wall_ratio": _median(paired_ratios),
            "median_application_overhead_fraction": round(overhead_fraction, 4)
            if overhead_fraction is not None
            else None,
        },
        "interpretation_note": (
            "Each round unloads the model, performs the ordinary preload, then performs an exact-runner warm-up "
            "using the same effective model, system prompt, context, output budget, temperature, num_thread and "
            "num_batch as measured requests. A round is excluded from the paired wall ratio if either measured "
            "request still reports material model load time."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 11 v4: exact-runner warm paired SovereignAI vs direct Ollama latency benchmark."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--keep-alive", default="60s")
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--max-measured-load-seconds", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=Path("workspace/e2e_latency_v4.json"))
    args = parser.parse_args()

    _require_local(args.api_url)
    _require_local(args.ollama_url)
    repeats = max(2, args.repeats)

    rounds: list[dict[str, Any]] = []
    chat_runs: list[dict[str, Any]] = []
    direct_runs: list[dict[str, Any]] = []
    paired_ratios: list[float] = []
    invalid_rounds: list[int] = []

    with httpx.Client(follow_redirects=False) as client:
        api_health = client.get(f"{args.api_url.rstrip('/')}/docs", timeout=5.0)
        if not api_health.is_success:
            raise RuntimeError(f"SovereignAI backend is not reachable at {args.api_url}")
        ollama_health = client.get(f"{args.ollama_url.rstrip('/')}/api/tags", timeout=5.0)
        ollama_health.raise_for_status()

        _prewarm_model(
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
                f"Prewarmed model {args.model} differs from routed model {direct_settings['effective_model']}"
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

            # The lightweight READY preload can use different runner options. Force the exact benchmark
            # runner configuration before either target is timed; this warm-up is deliberately unmeasured.
            exact_warm = _direct_ollama_stream(
                client,
                args.ollama_url,
                prompt=args.prompt,
                effective_model=direct_settings["effective_model"],
                system=SYSTEM_PROMPT,
                options=direct_settings["options"],
                keep_alive=args.keep_alive,
                timeout_seconds=args.timeout,
            )

            order = ["sovereignai", "direct_ollama"] if index % 2 == 0 else ["direct_ollama", "sovereignai"]
            pair: dict[str, Any] = {
                "round": index + 1,
                "order": order,
                "reset": reset,
                "exact_runner_warmup": exact_warm,
                "memory_before_pair": _memory_snapshot(),
            }
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
                        raise RuntimeError(f"Round {index + 1} /api/chat fell back")
                    chat_runs.append(run)
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

            measured_loads = {
                target: _measured_load_seconds(target, pair[target]) for target in ("sovereignai", "direct_ollama")
            }
            pair["measured_load_seconds"] = measured_loads
            valid_warm_path = all(
                value is not None and value <= args.max_measured_load_seconds
                for value in measured_loads.values()
            )
            pair["valid_warm_path"] = valid_warm_path

            sovereign_wall = float(pair["sovereignai"]["wall_seconds"])
            direct_wall = float(pair["direct_ollama"]["wall_seconds"])
            ratio = sovereign_wall / direct_wall if direct_wall > 0 else 0.0
            pair["sovereignai_vs_direct_wall_ratio"] = round(ratio, 4) if ratio else None
            pair["memory_after_pair"] = _memory_snapshot()
            rounds.append(pair)

            if valid_warm_path and ratio > 0:
                paired_ratios.append(ratio)
            else:
                invalid_rounds.append(index + 1)

            progress = _build_report(
                args=args,
                calibration=calibration,
                direct_settings=direct_settings,
                rounds=rounds,
                chat_runs=chat_runs,
                direct_runs=direct_runs,
                paired_ratios=paired_ratios,
                invalid_rounds=invalid_rounds,
                status="running" if len(rounds) < repeats else "complete",
            )
            _write_report(args.output, progress)

    status = "complete" if len(invalid_rounds) < repeats else "invalid"
    report = _build_report(
        args=args,
        calibration=calibration,
        direct_settings=direct_settings,
        rounds=rounds,
        chat_runs=chat_runs,
        direct_runs=direct_runs,
        paired_ratios=paired_ratios,
        invalid_rounds=invalid_rounds,
        status=status,
        error=("All rounds still incurred material model loading." if status == "invalid" else None),
    )
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    _write_report(args.output, report)
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
