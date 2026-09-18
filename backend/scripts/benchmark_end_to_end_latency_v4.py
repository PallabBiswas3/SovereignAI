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


def _reserve_from_calibration(calibration: dict[str, Any]) -> float:
    runtime = calibration.get("runtime_metrics") if isinstance(calibration.get("runtime_metrics"), dict) else {}
    provider = runtime.get("provider_generation") if isinstance(runtime.get("provider_generation"), dict) else {}
    admission = provider.get("resource_admission") if isinstance(provider.get("resource_admission"), dict) else {}
    value = admission.get("ram_reserve_mb")
    return float(value) if isinstance(value, (int, float)) else 0.0


def _wait_for_ram_headroom(*, minimum_available_mb: float, timeout_seconds: float) -> dict[str, Any]:
    """Wait for Windows/background-memory jitter to settle before a timed request.

    This does not alter SovereignAI's admission reserve. It only prevents a benchmark from starting
    while available memory is sitting exactly on the production safety boundary.
    """
    started = time.monotonic()
    best = _memory_snapshot()
    while time.monotonic() - started < timeout_seconds:
        current = _memory_snapshot()
        if current["available_mb"] > best["available_mb"]:
            best = current
        if current["available_mb"] >= minimum_available_mb:
            return {
                "ready": True,
                "wait_seconds": round(time.monotonic() - started, 3),
                "minimum_available_mb": round(minimum_available_mb, 2),
                "memory": current,
            }
        time.sleep(0.25)
    current = _memory_snapshot()
    return {
        "ready": current["available_mb"] >= minimum_available_mb,
        "wait_seconds": round(time.monotonic() - started, 3),
        "minimum_available_mb": round(minimum_available_mb, 2),
        "memory": current,
        "best_available_mb": best["available_mb"],
    }


def _exact_runner_warmup(
    client: httpx.Client,
    ollama_url: str,
    *,
    effective_model: str,
    options: dict[str, int | float],
    keep_alive: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Configure the exact runner with minimal prompt/output-cache growth.

    Context, thread count and batch size match the measured path. Only num_predict is reduced to 1
    because output length does not define the runner shape and a full benchmark answer would consume
    avoidable prompt/cache memory before timing begins.
    """
    warm_options = dict(options)
    warm_options["num_predict"] = 1
    payload: dict[str, Any] = {
        "model": effective_model,
        "prompt": "/no_think\nReply with one token: READY",
        "system": SYSTEM_PROMPT,
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "options": warm_options,
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
            f"Exact-runner warm-up failed with HTTP {response.status_code}: {response.text[:2000]}"
        )
    data = response.json()
    return {
        "wall_seconds": round(wall, 6),
        "load_duration_seconds": round(float(data.get("load_duration", 0) or 0) / 1_000_000_000, 6),
        "response": str(data.get("response") or "").strip(),
        "options": warm_options,
        "memory_after": _memory_snapshot(),
    }


def _build_report(
    *,
    args: argparse.Namespace,
    calibration: dict[str, Any],
    direct_settings: dict[str, Any],
    ram_reserve_mb: float,
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
        "benchmark_design": "paired-e2e-warm-path-v4-exact-runner-prewarm-ram-stable",
        "status": status,
        "error": error,
        "prompt": args.prompt,
        "requested_repeats": max(2, args.repeats),
        "completed_rounds": len(rounds),
        "invalid_warm_path_rounds": invalid_rounds,
        "max_measured_load_seconds": args.max_measured_load_seconds,
        "ram_reserve_mb": round(ram_reserve_mb, 2),
        "ram_headroom_mb": args.ram_headroom_mb,
        "minimum_benchmark_available_mb": round(ram_reserve_mb + args.ram_headroom_mb, 2),
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
            "Each round unloads/reloads the model, configures the exact runner with a one-token warm-up, and then "
            "waits for available RAM to exceed the unchanged SovereignAI reserve plus a benchmark-only headroom "
            "margin before timing requests. A round is excluded if either measured call still incurs material model load."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 11 v4: stable exact-runner warm SovereignAI vs direct Ollama latency benchmark."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--keep-alive", default="60s")
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--ram-headroom-mb", type=float, default=96.0)
    parser.add_argument("--ram-wait-seconds", type=float, default=20.0)
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
        ram_reserve_mb = _reserve_from_calibration(calibration)
        if ram_reserve_mb <= 0:
            raise RuntimeError("Calibration did not expose the SovereignAI RAM reserve")
        minimum_available_mb = ram_reserve_mb + max(0.0, args.ram_headroom_mb)

        for index in range(repeats):
            reset = _reset_and_rewarm(
                client,
                args.ollama_url,
                model=direct_settings["effective_model"],
                keep_alive=args.keep_alive,
                timeout_seconds=args.timeout,
                settle_seconds=max(0.0, args.settle_seconds),
            )
            exact_warm = _exact_runner_warmup(
                client,
                args.ollama_url,
                effective_model=direct_settings["effective_model"],
                options=direct_settings["options"],
                keep_alive=args.keep_alive,
                timeout_seconds=args.timeout,
            )
            pre_pair_ram = _wait_for_ram_headroom(
                minimum_available_mb=minimum_available_mb,
                timeout_seconds=max(0.0, args.ram_wait_seconds),
            )
            pair: dict[str, Any] = {
                "round": index + 1,
                "reset": reset,
                "exact_runner_warmup": exact_warm,
                "pre_pair_ram_gate": pre_pair_ram,
            }
            if not pre_pair_ram["ready"]:
                pair["valid_warm_path"] = False
                pair["error"] = (
                    f"RAM did not recover to benchmark threshold {minimum_available_mb:.0f} MB; "
                    f"available={pre_pair_ram['memory']['available_mb']:.0f} MB"
                )
                rounds.append(pair)
                invalid_rounds.append(index + 1)
                progress = _build_report(
                    args=args,
                    calibration=calibration,
                    direct_settings=direct_settings,
                    ram_reserve_mb=ram_reserve_mb,
                    rounds=rounds,
                    chat_runs=chat_runs,
                    direct_runs=direct_runs,
                    paired_ratios=paired_ratios,
                    invalid_rounds=invalid_rounds,
                    status="running" if len(rounds) < repeats else "complete",
                )
                _write_report(args.output, progress)
                continue

            order = ["sovereignai", "direct_ollama"] if index % 2 == 0 else ["direct_ollama", "sovereignai"]
            pair["order"] = order
            pair["memory_before_pair"] = _memory_snapshot()

            failed = False
            for target in order:
                target_ram = _wait_for_ram_headroom(
                    minimum_available_mb=minimum_available_mb,
                    timeout_seconds=max(0.0, args.ram_wait_seconds),
                )
                pair[f"ram_gate_before_{target}"] = target_ram
                if not target_ram["ready"]:
                    pair["error"] = (
                        f"RAM gate failed before {target}: "
                        f"available={target_ram['memory']['available_mb']:.0f} MB, "
                        f"required={minimum_available_mb:.0f} MB"
                    )
                    failed = True
                    break
                try:
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
                except RuntimeError as exc:
                    pair["error"] = str(exc)
                    failed = True
                    break

            if failed or "sovereignai" not in pair or "direct_ollama" not in pair:
                pair["valid_warm_path"] = False
                pair["memory_after_pair"] = _memory_snapshot()
                rounds.append(pair)
                invalid_rounds.append(index + 1)
                progress = _build_report(
                    args=args,
                    calibration=calibration,
                    direct_settings=direct_settings,
                    ram_reserve_mb=ram_reserve_mb,
                    rounds=rounds,
                    chat_runs=chat_runs,
                    direct_runs=direct_runs,
                    paired_ratios=paired_ratios,
                    invalid_rounds=invalid_rounds,
                    status="running" if len(rounds) < repeats else "complete",
                )
                _write_report(args.output, progress)
                continue

            measured_loads = {
                target: _measured_load_seconds(target, pair[target])
                for target in ("sovereignai", "direct_ollama")
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
                ram_reserve_mb=ram_reserve_mb,
                rounds=rounds,
                chat_runs=chat_runs,
                direct_runs=direct_runs,
                paired_ratios=paired_ratios,
                invalid_rounds=invalid_rounds,
                status="running" if len(rounds) < repeats else "complete",
            )
            _write_report(args.output, progress)

    valid_rounds = repeats - len(invalid_rounds)
    status = "complete" if valid_rounds >= max(2, repeats // 2) else "insufficient-valid-rounds"
    error = None if status == "complete" else f"Only {valid_rounds} of {repeats} rounds were valid."
    report = _build_report(
        args=args,
        calibration=calibration,
        direct_settings=direct_settings,
        ram_reserve_mb=ram_reserve_mb,
        rounds=rounds,
        chat_runs=chat_runs,
        direct_runs=direct_runs,
        paired_ratios=paired_ratios,
        invalid_rounds=invalid_rounds,
        status=status,
        error=error,
    )
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    _write_report(args.output, report)
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
