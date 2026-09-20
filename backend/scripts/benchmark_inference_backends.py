from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import httpx

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.resources.backend_selection import BackendSelectionProfile, BackendSelectionStore
from app.resources.cpu_tuning import CpuTuningStore

QUALITY_CASES = [
    {"name": "arithmetic", "prompt": "Return only the integer result of 17 * 19.", "contains": ["323"]},
    {
        "name": "industrial_extraction",
        "prompt": (
            "Read this maintenance note: Pump P-102 has RMS vibration 8.2 mm/s, "
            "bearing temperature 71 C, and discharge pressure 4.6 bar. "
            "Return only the vibration value with its unit."
        ),
        "contains": ["8.2", "mm/s"],
    },
    {"name": "instruction_following", "prompt": "Output exactly this text and nothing else: SAFE_LOCAL_ONLY", "exact": "SAFE_LOCAL_ONLY"},
    {"name": "numeric_comparison", "prompt": "Which number is larger, 0.875 or 0.857? Return only the larger number.", "contains": ["0.875"]},
]

PERF_PROMPT = "Explain in four concise technical points why an outer-race bearing fault creates periodic vibration impulses."


def _quality_pass(case: dict[str, Any], response: str) -> bool:
    normalized = " ".join(response.strip().split())
    if "exact" in case:
        return normalized == str(case["exact"])
    lowered = normalized.lower()
    return all(str(value).lower() in lowered for value in case.get("contains", []))


def _ollama_generate(
    client: httpx.Client,
    endpoint: str,
    model: str,
    prompt: str,
    *,
    num_ctx: int,
    num_predict: int,
    num_thread: int | None,
    num_batch: int | None,
) -> dict[str, Any]:
    options: dict[str, int | float] = {
        "temperature": 0.0,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
    }
    if num_thread:
        options["num_thread"] = num_thread
    if num_batch:
        options["num_batch"] = num_batch
    started = time.perf_counter()
    response = client.post(
        f"{endpoint}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "60s",
            "options": options,
        },
    )
    wall = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()
    eval_duration = float(data.get("eval_duration", 0) or 0) / 1_000_000_000
    eval_count = int(data.get("eval_count", 0) or 0)
    return {
        "response": str(data.get("response", "")).strip(),
        "wall_seconds": round(wall, 6),
        "tokens_per_second": round(eval_count / eval_duration, 3) if eval_duration and eval_count else None,
        "generated_tokens": eval_count,
        "prompt_tokens": int(data.get("prompt_eval_count", 0) or 0),
        "done_reason": str(data.get("done_reason") or "stop"),
    }


def _llama_cpp_generate(
    client: httpx.Client,
    endpoint: str,
    model: str,
    prompt: str,
    *,
    num_predict: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.post(
        f"{endpoint}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": num_predict,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        },
    )
    wall = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices", [])
    message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
    timings = data.get("timings") if isinstance(data.get("timings"), dict) else {}
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    tps = timings.get("predicted_per_second")
    return {
        "response": str(message.get("content") or "").strip(),
        "wall_seconds": round(wall, 6),
        "tokens_per_second": round(float(tps), 3) if isinstance(tps, (int, float)) else None,
        "generated_tokens": int(timings.get("predicted_n") or usage.get("completion_tokens") or 0),
        "prompt_tokens": int((timings.get("prompt_n") or 0) + (timings.get("cache_n") or 0) or usage.get("prompt_tokens") or 0),
        "done_reason": str(choices[0].get("finish_reason") or "stop") if choices and isinstance(choices[0], dict) else "stop",
    }


def _backend_quality(run, client: httpx.Client, endpoint: str, model: str, **kwargs: Any) -> tuple[float, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for case in QUALITY_CASES:
        result = run(client, endpoint, model, str(case["prompt"]), num_predict=64, **kwargs)
        passed = _quality_pass(case, result["response"])
        records.append({"name": case["name"], "passed": passed, "response": result["response"]})
    score = sum(1 for item in records if item["passed"]) / len(records)
    return round(score, 3), records


def _performance(run, client: httpx.Client, endpoint: str, model: str, *, repeats: int, **kwargs: Any) -> list[dict[str, Any]]:
    run(client, endpoint, model, PERF_PROMPT, num_predict=128, **kwargs)
    return [run(client, endpoint, model, PERF_PROMPT, num_predict=128, **kwargs) for _ in range(max(1, repeats))]


def _summary(runs: list[dict[str, Any]]) -> dict[str, float | None]:
    tps = [float(item["tokens_per_second"]) for item in runs if item.get("tokens_per_second") is not None]
    wall = [float(item["wall_seconds"]) for item in runs]
    return {
        "median_tokens_per_second": round(median(tps), 3) if tps else None,
        "median_wall_seconds": round(median(wall), 6) if wall else None,
    }


def _paired_interleaved_performance(
    client: httpx.Client,
    *,
    ollama_endpoint: str,
    llama_endpoint: str,
    ollama_model: str,
    llama_model: str,
    repeats: int,
    num_ctx: int,
    num_thread: int | None,
    num_batch: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], float | None]:
    """Measure paired A/B rounds while alternating request order to reduce thermal/order bias."""
    # Excluded warmups put both runners into their normal steady-state paths before measured rounds.
    _ollama_generate(
        client, ollama_endpoint, ollama_model, PERF_PROMPT,
        num_ctx=num_ctx, num_predict=128, num_thread=num_thread, num_batch=num_batch,
    )
    _llama_cpp_generate(client, llama_endpoint, llama_model, PERF_PROMPT, num_predict=128)

    ollama_runs: list[dict[str, Any]] = []
    llama_runs: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    ratios: list[float] = []
    for index in range(max(2, repeats)):
        order = ("ollama", "llama_cpp") if index % 2 == 0 else ("llama_cpp", "ollama")
        results: dict[str, dict[str, Any]] = {}
        for backend in order:
            if backend == "ollama":
                results[backend] = _ollama_generate(
                    client, ollama_endpoint, ollama_model, PERF_PROMPT,
                    num_ctx=num_ctx, num_predict=128,
                    num_thread=num_thread, num_batch=num_batch,
                )
            else:
                results[backend] = _llama_cpp_generate(
                    client, llama_endpoint, llama_model, PERF_PROMPT, num_predict=128,
                )
        ollama_run = results["ollama"]
        llama_run = results["llama_cpp"]
        ollama_runs.append(ollama_run)
        llama_runs.append(llama_run)
        ollama_tps = float(ollama_run.get("tokens_per_second") or 0.0)
        llama_tps = float(llama_run.get("tokens_per_second") or 0.0)
        ratio = llama_tps / ollama_tps if ollama_tps > 0 and llama_tps > 0 else None
        if ratio is not None:
            ratios.append(ratio)
        rounds.append({
            "round": index + 1,
            "order": list(order),
            "ollama_tokens_per_second": ollama_run.get("tokens_per_second"),
            "llama_cpp_tokens_per_second": llama_run.get("tokens_per_second"),
            "llama_cpp_vs_ollama_ratio": round(ratio, 4) if ratio is not None else None,
        })
    paired_ratio = round(median(ratios), 4) if ratios else None
    return ollama_runs, llama_runs, rounds, paired_ratio


def _llama_server_identity(client: httpx.Client, endpoint: str) -> list[str] | None:
    try:
        health = client.get(f"{endpoint}/health", timeout=2.0)
        health.raise_for_status()
        models_response = client.get(f"{endpoint}/v1/models", timeout=2.0)
        models_response.raise_for_status()
        models = [item for item in models_response.json().get("data", []) if isinstance(item, dict)]
        if not models:
            return None
        return [str(item.get("id") or "") for item in models]
    except (httpx.HTTPError, ValueError):
        return None


def _discover_llama_endpoint(
    client: httpx.Client,
    requested_endpoint: str,
    expected_model: str,
    *,
    explicit_endpoint: bool,
) -> tuple[str, list[str] | None]:
    requested = requested_endpoint.rstrip("/")
    ids = _llama_server_identity(client, requested)
    if ids is not None:
        return requested, ids
    if explicit_endpoint:
        return requested, None

    for port in range(8081, 8101):
        candidate = f"http://127.0.0.1:{port}"
        ids = _llama_server_identity(client, candidate)
        if ids is not None and expected_model in ids:
            return candidate, ids
    return requested, None


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Compare Ollama and llama.cpp on the same local model and persist the validated backend winner.")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--ollama-endpoint", default=settings.ollama_url)
    parser.add_argument("--llama-cpp-endpoint", default=settings.llama_cpp_url)
    parser.add_argument("--llama-cpp-model", default=None, help="Model id reported by llama.cpp /v1/models. For automatic promotion it must equal --model.")
    parser.add_argument("--num-ctx", type=int, default=settings.llama_cpp_context_size)
    parser.add_argument("--repeats", type=int, default=4, help="Paired interleaved A/B rounds; alternating order reduces thermal bias.")
    parser.add_argument("--min-quality", type=float, default=settings.model_optimization_min_quality)
    parser.add_argument("--min-speedup", type=float, default=settings.model_backend_min_speedup)
    parser.add_argument("--output", type=Path, default=Path("workspace/backend_comparison.json"))
    parser.add_argument("--no-save-profile", action="store_true")
    args = parser.parse_args()

    ollama = args.ollama_endpoint.rstrip("/")
    llama_requested = args.llama_cpp_endpoint.rstrip("/")
    tuning = CpuTuningStore(settings.model_cpu_tuning_path).get(args.model)
    thread = tuning.num_thread if tuning else None
    batch = tuning.num_batch if tuning else None

    with httpx.Client(timeout=httpx.Timeout(settings.model_generation_timeout_seconds)) as client:
        llama, discovered_ids = _discover_llama_endpoint(
            client,
            llama_requested,
            args.model,
            explicit_endpoint="--llama-cpp-endpoint" in sys.argv,
        )

        tags = client.get(f"{ollama}/api/tags")
        tags.raise_for_status()
        installed = {str(item.get("name") or item.get("model") or "") for item in tags.json().get("models", []) if isinstance(item, dict)}
        if args.model not in installed:
            raise RuntimeError(f"Ollama model '{args.model}' is not installed")

        ollama_quality, ollama_cases = _backend_quality(
            _ollama_generate, client, ollama, args.model,
            num_ctx=args.num_ctx, num_thread=thread, num_batch=batch,
        )

        llama_available = False
        llama_identity_matches = False
        llama_error: str | None = None
        llama_model = args.llama_cpp_model
        llama_quality = 0.0
        llama_cases: list[dict[str, Any]] = []
        llama_runs: list[dict[str, Any]] = []
        ollama_runs: list[dict[str, Any]] = []
        interleaved_rounds: list[dict[str, Any]] = []
        paired_speed_ratio: float | None = None
        try:
            reported_ids = discovered_ids or _llama_server_identity(client, llama)
            if not reported_ids:
                raise RuntimeError(f"No compatible llama.cpp server found at {llama_requested} or localhost ports 8081-8100")
            if llama_model is None:
                llama_model = args.model if args.model in reported_ids else reported_ids[0]
            llama_identity_matches = llama_model == args.model and llama_model in reported_ids
            llama_quality, llama_cases = _backend_quality(_llama_cpp_generate, client, llama, llama_model)
            llama_available = True
            ollama_runs, llama_runs, interleaved_rounds, paired_speed_ratio = _paired_interleaved_performance(
                client,
                ollama_endpoint=ollama,
                llama_endpoint=llama,
                ollama_model=args.model,
                llama_model=llama_model,
                repeats=args.repeats,
                num_ctx=args.num_ctx,
                num_thread=thread,
                num_batch=batch,
            )
        except Exception as exc:
            llama_error = str(exc)
            ollama_runs = _performance(
                _ollama_generate, client, ollama, args.model,
                repeats=args.repeats, num_ctx=args.num_ctx,
                num_thread=thread, num_batch=batch,
            )

    ollama_summary = _summary(ollama_runs)
    llama_summary = _summary(llama_runs) if llama_runs else {"median_tokens_per_second": None, "median_wall_seconds": None}
    ollama_tps = float(ollama_summary["median_tokens_per_second"] or 0.0)
    llama_tps = float(llama_summary["median_tokens_per_second"] or 0.0)
    median_speedup = llama_tps / ollama_tps if ollama_tps > 0 and llama_tps > 0 else 0.0
    decision_speedup = paired_speed_ratio if paired_speed_ratio is not None else median_speedup
    quality_floor = max(float(args.min_quality), ollama_quality - 0.05)
    use_llama = (
        llama_available
        and llama_identity_matches
        and llama_quality >= quality_floor
        and decision_speedup >= float(args.min_speedup)
    )
    selected_backend = "llama_cpp" if use_llama else "ollama"
    selected_endpoint = llama if use_llama else ollama
    selected_quality = llama_quality if use_llama else ollama_quality
    selected_summary = llama_summary if use_llama else ollama_summary

    profile = BackendSelectionProfile(
        model=args.model,
        backend=selected_backend,
        endpoint=selected_endpoint,
        backend_model=(llama_model if use_llama else args.model),
        quality_score=selected_quality,
        median_tokens_per_second=(float(selected_summary["median_tokens_per_second"]) if selected_summary["median_tokens_per_second"] else None),
        median_wall_seconds=(float(selected_summary["median_wall_seconds"]) if selected_summary["median_wall_seconds"] else None),
        speedup_vs_ollama=(decision_speedup if use_llama else 1.0),
    )
    if not args.no_save_profile:
        BackendSelectionStore(settings.model_backend_selection_path).save(profile)

    report = {
        "model": args.model,
        "benchmark_design": "paired-interleaved-v2",
        "llama_cpp_endpoint_requested": llama_requested,
        "llama_cpp_endpoint_used": llama,
        "llama_cpp_model": llama_model,
        "llama_cpp_identity_matches": llama_identity_matches,
        "selection_rules": {
            "same_logical_model_required": True,
            "minimum_quality": args.min_quality,
            "quality_floor_vs_ollama": round(quality_floor, 3),
            "minimum_speedup": args.min_speedup,
            "decision_metric": "median paired llama_cpp/ollama throughput ratio",
        },
        "ollama": {"quality_score": ollama_quality, "quality_cases": ollama_cases, "summary": ollama_summary, "runs": ollama_runs},
        "llama_cpp": {
            "available": llama_available,
            "error": llama_error,
            "quality_score": llama_quality,
            "quality_cases": llama_cases,
            "summary": llama_summary,
            "runs": llama_runs,
        },
        "interleaved_rounds": interleaved_rounds,
        "median_speedup_from_backend_medians": round(median_speedup, 4) if median_speedup else None,
        "paired_speed_ratio_llama_vs_ollama": paired_speed_ratio,
        "selected": profile.model_dump(mode="json"),
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
