from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import httpx
import psutil

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.resources.cpu_tuning import CpuTuningStore
from app.resources.kv_cache_optimization import KvCacheOptimizationProfile, KvCacheOptimizationStore

MIB = 1024 * 1024
CACHE_TYPES = ("f16", "q8_0", "q4_0")
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
    {
        "name": "long_context_retrieval",
        "prompt": (
            "Ignore all distractors and return only the code after TARGET=.\n"
            + "\n".join(f"distractor_{i}=value_{i}" for i in range(220))
            + "\nTARGET=PUMP_SAFE_7421\nReturn only the target code."
        ),
        "exact": "PUMP_SAFE_7421",
    },
]
PERF_PROMPT = "Explain in four concise technical points why an outer-race bearing fault creates periodic vibration impulses."


def memory_snapshot() -> dict[str, float]:
    mem = psutil.virtual_memory()
    return {
        "available_mb": round(mem.available / MIB, 2),
        "used_mb": round(mem.used / MIB, 2),
        "percent": round(float(mem.percent), 2),
    }


def free_port(start: int) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free localhost port found for isolated Ollama benchmark")


def quality_pass(case: dict[str, Any], response: str) -> bool:
    normalized = " ".join(response.strip().split())
    if "exact" in case:
        return normalized == str(case["exact"])
    lowered = normalized.lower()
    return all(str(value).lower() in lowered for value in case.get("contains", []))


def wait_ready(client: httpx.Client, endpoint: str, process: subprocess.Popen[Any], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"isolated Ollama server exited early with code {process.returncode}")
        try:
            response = client.get(f"{endpoint}/api/tags", timeout=1.0)
            if response.is_success:
                return
            last_error = f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"isolated Ollama server did not become ready: {last_error}")


def unload(client: httpx.Client, endpoint: str, model: str) -> None:
    try:
        client.post(f"{endpoint}/api/generate", json={"model": model, "stream": False, "keep_alive": 0})
    except httpx.HTTPError:
        pass


def generate(
    client: httpx.Client,
    endpoint: str,
    *,
    model: str,
    prompt: str,
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
    eval_seconds = float(data.get("eval_duration", 0) or 0) / 1_000_000_000
    eval_count = int(data.get("eval_count", 0) or 0)
    return {
        "response": str(data.get("response", "")).strip(),
        "wall_seconds": round(wall, 6),
        "generated_tokens": eval_count,
        "tokens_per_second": round(eval_count / eval_seconds, 3) if eval_seconds and eval_count else None,
        "prompt_tokens": int(data.get("prompt_eval_count", 0) or 0),
        "done_reason": str(data.get("done_reason") or "stop"),
    }


def resident_model(client: httpx.Client, endpoint: str, model: str) -> dict[str, Any] | None:
    response = client.get(f"{endpoint}/api/ps")
    response.raise_for_status()
    names = {model, f"{model}:latest"} if ":" not in model else {model}
    for item in response.json().get("models", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("name") or item.get("model") or "") in names:
            return item
    return None


def benchmark_cache(
    *,
    ollama_exe: str,
    model: str,
    cache_type: str,
    port: int,
    num_ctx: int,
    repeats: int,
    num_thread: int | None,
    num_batch: int | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    endpoint = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    env["OLLAMA_FLASH_ATTENTION"] = "1"
    env["OLLAMA_KV_CACHE_TYPE"] = cache_type
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        [ollama_exe, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )
    before = memory_snapshot()
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_seconds)) as client:
            wait_ready(client, endpoint, process)
            tags = client.get(f"{endpoint}/api/tags")
            tags.raise_for_status()
            names = {str(item.get("name") or item.get("model") or "") for item in tags.json().get("models", []) if isinstance(item, dict)}
            if model not in names:
                raise RuntimeError(f"Model '{model}' is not visible to isolated Ollama server on port {port}")

            # Force the configured context/KV allocation before RAM measurement.
            generate(
                client, endpoint, model=model, prompt=PERF_PROMPT,
                num_ctx=num_ctx, num_predict=64, num_thread=num_thread, num_batch=num_batch,
            )
            after_load = memory_snapshot()
            resident = resident_model(client, endpoint, model)

            quality_runs: list[dict[str, Any]] = []
            for case in QUALITY_CASES:
                result = generate(
                    client, endpoint, model=model, prompt=str(case["prompt"]),
                    num_ctx=num_ctx, num_predict=64, num_thread=num_thread, num_batch=num_batch,
                )
                quality_runs.append({
                    "name": case["name"],
                    "passed": quality_pass(case, result["response"]),
                    "response": result["response"],
                    "wall_seconds": result["wall_seconds"],
                })
            quality_score = sum(1 for item in quality_runs if item["passed"]) / len(quality_runs)

            perf_runs = [
                generate(
                    client, endpoint, model=model, prompt=PERF_PROMPT,
                    num_ctx=num_ctx, num_predict=128, num_thread=num_thread, num_batch=num_batch,
                )
                for _ in range(max(2, repeats))
            ]
            tps = [float(item["tokens_per_second"]) for item in perf_runs if item.get("tokens_per_second") is not None]
            wall = [float(item["wall_seconds"]) for item in perf_runs]
            after_benchmark = memory_snapshot()
            return {
                "cache_type": cache_type,
                "endpoint": endpoint,
                "quality_score": round(quality_score, 3),
                "quality_runs": quality_runs,
                "median_tokens_per_second": round(median(tps), 3) if tps else None,
                "mean_tokens_per_second": round(mean(tps), 3) if tps else None,
                "median_wall_seconds": round(median(wall), 6),
                "resident_size_mb": round(float(resident.get("size_vram", 0) or resident.get("size", 0)) / MIB, 2) if resident else None,
                "resident_context_length": int(resident.get("context_length", 0) or 0) if resident else None,
                "system_ram_delta_mb": round(max(0.0, before["available_mb"] - after_load["available_mb"]), 2),
                "available_ram_before_mb": before["available_mb"],
                "available_ram_after_load_mb": after_load["available_mb"],
                "available_ram_after_benchmark_mb": after_benchmark["available_mb"],
                "performance_runs": perf_runs,
            }
    finally:
        try:
            with httpx.Client(timeout=3.0) as cleanup:
                unload(cleanup, endpoint, model)
        except Exception:
            pass
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        time.sleep(1.0)


def select_result(
    results: list[dict[str, Any]],
    *,
    min_quality: float,
    max_tps_regression: float,
    min_ram_saving_mb: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = next(item for item in results if item["cache_type"] == "f16")
    base_tps = float(baseline.get("median_tokens_per_second") or 0.0)
    base_ram = float(baseline.get("system_ram_delta_mb") or 0.0)
    eligible: list[tuple[float, float, int, dict[str, Any]]] = []
    rank = {"f16": 0, "q8_0": 1, "q4_0": 2}
    for item in results:
        tps = float(item.get("median_tokens_per_second") or 0.0)
        ram = float(item.get("system_ram_delta_mb") or 0.0)
        quality = float(item.get("quality_score") or 0.0)
        tps_ratio = tps / base_tps if base_tps > 0 and tps > 0 else 0.0
        ram_saving = max(0.0, base_ram - ram)
        item["tps_ratio_vs_f16"] = round(tps_ratio, 4) if tps_ratio else None
        item["ram_saving_vs_f16_mb"] = round(ram_saving, 2)
        qualifies = (
            item["cache_type"] == "f16"
            or (
                quality >= min_quality
                and tps_ratio >= 1.0 - max_tps_regression
                and ram_saving >= min_ram_saving_mb
            )
        )
        item["eligible"] = qualifies
        if qualifies:
            eligible.append((ram_saving, tps_ratio, rank[item["cache_type"]], item))
    # Prefer meaningful RAM saving first, then throughput, then the less aggressive cache type on ties.
    selected = max(eligible, key=lambda value: (value[0], value[1], -value[2]))[3]
    policy = {
        "minimum_quality": min_quality,
        "maximum_tps_regression": max_tps_regression,
        "minimum_ram_saving_mb": min_ram_saving_mb,
        "selection_priority": "maximize RAM saving subject to quality and throughput guardrails",
    }
    return selected, policy


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Benchmark Ollama f16/q8_0/q4_0 KV cache modes in isolated local server processes.")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--contexts", default="2048,4096", help="Comma-separated context sizes. Each cache type is benchmarked independently per context.")
    parser.add_argument("--cache-types", default="f16,q8_0,q4_0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--base-port", type=int, default=11600)
    parser.add_argument("--ollama-exe", default=None)
    parser.add_argument("--min-quality", type=float, default=settings.model_kv_cache_min_quality)
    parser.add_argument("--max-tps-regression", type=float, default=settings.model_kv_cache_max_tps_regression)
    parser.add_argument("--min-ram-saving-mb", type=float, default=settings.model_kv_cache_min_ram_saving_mb)
    parser.add_argument("--output", type=Path, default=Path("workspace/kv_cache_benchmark.json"))
    parser.add_argument("--no-save-profile", action="store_true")
    args = parser.parse_args()

    ollama_exe = args.ollama_exe or shutil.which("ollama")
    if not ollama_exe:
        raise RuntimeError("Could not find 'ollama' executable in PATH. Pass --ollama-exe explicitly.")
    cache_types = [item.strip().lower() for item in args.cache_types.split(",") if item.strip()]
    invalid = [item for item in cache_types if item not in CACHE_TYPES]
    if invalid:
        raise ValueError(f"Unsupported cache type(s): {', '.join(invalid)}")
    contexts = sorted({int(item.strip()) for item in args.contexts.split(",") if item.strip()})
    if not contexts:
        raise ValueError("At least one context size is required")
    tuning = CpuTuningStore(settings.model_cpu_tuning_path).get(args.model)
    num_thread = tuning.num_thread if tuning else None
    num_batch = tuning.num_batch if tuning else None

    all_contexts: list[dict[str, Any]] = []
    chosen_profiles: list[dict[str, Any]] = []
    next_port = args.base_port
    for context in contexts:
        results: list[dict[str, Any]] = []
        for cache_type in cache_types:
            port = free_port(next_port)
            next_port = port + 1
            results.append(benchmark_cache(
                ollama_exe=ollama_exe,
                model=args.model,
                cache_type=cache_type,
                port=port,
                num_ctx=context,
                repeats=args.repeats,
                num_thread=num_thread,
                num_batch=num_batch,
                timeout_seconds=settings.model_generation_timeout_seconds,
            ))
        selected, policy = select_result(
            results,
            min_quality=args.min_quality,
            max_tps_regression=args.max_tps_regression,
            min_ram_saving_mb=args.min_ram_saving_mb,
        )
        all_contexts.append({"num_ctx": context, "results": results, "selected": selected, "selection_policy": policy})
        chosen_profiles.append({"num_ctx": context, **selected})

    # Persist the preferred short-context profile. Runtime adaptive context still controls when larger contexts are needed.
    preferred_context = min(contexts)
    preferred = next(item for item in chosen_profiles if item["num_ctx"] == preferred_context)
    profile = KvCacheOptimizationProfile(
        model=args.model,
        cache_type=preferred["cache_type"],
        flash_attention=True,
        context_length=preferred_context,
        quality_score=float(preferred["quality_score"]),
        median_tokens_per_second=float(preferred["median_tokens_per_second"]) if preferred.get("median_tokens_per_second") else None,
        resident_size_mb=float(preferred["resident_size_mb"]) if preferred.get("resident_size_mb") else None,
        system_ram_delta_mb=float(preferred["system_ram_delta_mb"]) if preferred.get("system_ram_delta_mb") is not None else None,
        available_ram_after_mb=float(preferred["available_ram_after_load_mb"]) if preferred.get("available_ram_after_load_mb") is not None else None,
    )
    if not args.no_save_profile:
        KvCacheOptimizationStore(settings.model_kv_cache_optimization_path).save(profile)

    report = {
        "model": args.model,
        "ollama_executable": ollama_exe,
        "cpu_tuning": tuning.model_dump(mode="json") if tuning else None,
        "benchmark_design": "isolated-server-per-cache-v1",
        "contexts": all_contexts,
        "saved_profile": profile.model_dump(mode="json"),
        "runtime_note": "The selected cache type is an Ollama server environment setting. Use start_ollama_optimized.ps1 to launch Ollama with the persisted profile.",
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
