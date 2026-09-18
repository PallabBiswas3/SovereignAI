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
from statistics import median
from typing import Any

import httpx
import psutil

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.resources.cpu_tuning import CpuTuningStore
from app.resources.kv_cache_optimization import (
    KvCacheConfirmationStore,
    evaluate_context_confirmation,
)

MIB = 1024 * 1024
PERF_PROMPT = "Explain in four concise technical points why an outer-race bearing fault creates periodic vibration impulses."
SHORT_QUALITY_CASES = [
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
LONG_CONTEXT_CASE = {
    "name": "long_context_retrieval",
    "prompt": (
        "Ignore all distractors and return only the code after TARGET=.\n"
        + "\n".join(f"distractor_{i}=value_{i}" for i in range(220))
        + "\nTARGET=PUMP_SAFE_7421\nReturn only the target code."
    ),
    "exact": "PUMP_SAFE_7421",
}


def _quality_pass(case: dict[str, Any], response: str) -> bool:
    normalized = " ".join(response.strip().split())
    if "exact" in case:
        return normalized == str(case["exact"])
    lowered = normalized.lower()
    return all(str(value).lower() in lowered for value in case.get("contains", []))


def _memory_snapshot() -> dict[str, float]:
    mem = psutil.virtual_memory()
    return {
        "available_mb": round(mem.available / MIB, 2),
        "used_mb": round(mem.used / MIB, 2),
        "percent": round(float(mem.percent), 2),
    }


def _free_port(start: int) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free localhost port found")


def _wait_ready(client: httpx.Client, endpoint: str, process: subprocess.Popen[Any], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Ollama server exited early with code {process.returncode}")
        try:
            response = client.get(f"{endpoint}/api/tags", timeout=1.0)
            if response.is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"Ollama server did not become ready at {endpoint}")


def _generate(
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
    options: dict[str, int | float] = {"temperature": 0.0, "num_ctx": num_ctx, "num_predict": num_predict}
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
        "tokens_per_second": round(eval_count / eval_seconds, 3) if eval_seconds and eval_count else None,
        "generated_tokens": eval_count,
        "prompt_tokens": int(data.get("prompt_eval_count", 0) or 0),
    }


def _resident_model(client: httpx.Client, endpoint: str, model: str) -> dict[str, Any] | None:
    response = client.get(f"{endpoint}/api/ps")
    response.raise_for_status()
    for item in response.json().get("models", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("name") or item.get("model") or "") == model:
            return item
    return None


def _run_one(
    *,
    ollama_exe: str,
    model: str,
    cache_type: str,
    port: int,
    num_ctx: int,
    num_thread: int | None,
    num_batch: int | None,
    timeout_seconds: float,
    run_long_context: bool,
) -> dict[str, Any]:
    endpoint = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    env["OLLAMA_FLASH_ATTENTION"] = "1"
    env["OLLAMA_KV_CACHE_TYPE"] = cache_type
    process = subprocess.Popen(
        [ollama_exe, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    before = _memory_snapshot()
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_seconds)) as client:
            _wait_ready(client, endpoint, process)
            _generate(
                client, endpoint, model=model, prompt=PERF_PROMPT,
                num_ctx=num_ctx, num_predict=64, num_thread=num_thread, num_batch=num_batch,
            )
            after_load = _memory_snapshot()
            resident = _resident_model(client, endpoint, model)

            # Performance is deliberately measured before the expensive long-context canary.
            perf = _generate(
                client, endpoint, model=model, prompt=PERF_PROMPT,
                num_ctx=num_ctx, num_predict=128, num_thread=num_thread, num_batch=num_batch,
            )

            quality_runs: list[dict[str, Any]] = []
            for case in SHORT_QUALITY_CASES:
                result = _generate(
                    client, endpoint, model=model, prompt=str(case["prompt"]),
                    num_ctx=num_ctx, num_predict=64, num_thread=num_thread, num_batch=num_batch,
                )
                quality_runs.append({
                    "name": case["name"],
                    "passed": _quality_pass(case, result["response"]),
                    "response": result["response"],
                    "wall_seconds": result["wall_seconds"],
                })
            short_quality_score = sum(1 for item in quality_runs if item["passed"]) / len(quality_runs)

            long_context_result = None
            if run_long_context:
                result = _generate(
                    client, endpoint, model=model, prompt=str(LONG_CONTEXT_CASE["prompt"]),
                    num_ctx=num_ctx, num_predict=64, num_thread=num_thread, num_batch=num_batch,
                )
                long_context_result = {
                    "passed": _quality_pass(LONG_CONTEXT_CASE, result["response"]),
                    "response": result["response"],
                    "wall_seconds": result["wall_seconds"],
                }

            return {
                "cache_type": cache_type,
                "endpoint": endpoint,
                "memory_before": before,
                "memory_after_load": after_load,
                "system_ram_delta_mb": round(max(0.0, before["available_mb"] - after_load["available_mb"]), 2),
                "resident_size_mb": round(float(resident.get("size_vram", 0) or resident.get("size", 0)) / MIB, 2) if resident else None,
                "resident_context_length": int(resident.get("context_length", 0) or 0) if resident else None,
                "performance": perf,
                "short_quality_score": round(short_quality_score, 3),
                "quality_runs": quality_runs,
                "long_context": long_context_result,
            }
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        time.sleep(1.0)


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Controlled paired confirmation of f16 vs q8_0 KV cache at one context size.")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=11700)
    parser.add_argument("--ollama-exe", default=None)
    parser.add_argument("--min-quality", type=float, default=0.95)
    parser.add_argument("--min-speed-ratio", type=float, default=1.03)
    parser.add_argument("--min-win-rate", type=float, default=0.75)
    parser.add_argument("--min-resident-saving-mb", type=float, default=128.0)
    parser.add_argument("--run-long-context", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("workspace/kv_cache_confirmation_4096.json"))
    parser.add_argument("--confirmation-path", type=Path, default=Path("backend/data/kv_cache_confirmations.json"))
    args = parser.parse_args()

    ollama_exe = args.ollama_exe or shutil.which("ollama")
    if not ollama_exe:
        raise RuntimeError("Could not find ollama executable")
    tuning = CpuTuningStore(settings.model_cpu_tuning_path).get(args.model)
    num_thread = tuning.num_thread if tuning else None
    num_batch = tuning.num_batch if tuning else None

    rounds: list[dict[str, Any]] = []
    ratios: list[float] = []
    f16_sizes: list[float] = []
    q8_sizes: list[float] = []
    f16_quality_scores: list[float] = []
    q8_quality_scores: list[float] = []
    f16_long_pass: list[bool] = []
    q8_long_pass: list[bool] = []
    next_port = args.base_port

    for index in range(max(2, args.rounds)):
        order = ["f16", "q8_0"] if index % 2 == 0 else ["q8_0", "f16"]
        results: dict[str, dict[str, Any]] = {}
        for cache_type in order:
            port = _free_port(next_port)
            next_port = port + 1
            results[cache_type] = _run_one(
                ollama_exe=ollama_exe,
                model=args.model,
                cache_type=cache_type,
                port=port,
                num_ctx=args.num_ctx,
                num_thread=num_thread,
                num_batch=num_batch,
                timeout_seconds=settings.model_generation_timeout_seconds,
                run_long_context=args.run_long_context,
            )

        f16 = results["f16"]
        q8 = results["q8_0"]
        f16_tps = float(f16["performance"].get("tokens_per_second") or 0.0)
        q8_tps = float(q8["performance"].get("tokens_per_second") or 0.0)
        ratio = q8_tps / f16_tps if f16_tps > 0 and q8_tps > 0 else 0.0
        if ratio > 0:
            ratios.append(ratio)
        if f16.get("resident_size_mb"):
            f16_sizes.append(float(f16["resident_size_mb"]))
        if q8.get("resident_size_mb"):
            q8_sizes.append(float(q8["resident_size_mb"]))
        f16_quality_scores.append(float(f16["short_quality_score"]))
        q8_quality_scores.append(float(q8["short_quality_score"]))
        if args.run_long_context:
            f16_long_pass.append(bool((f16.get("long_context") or {}).get("passed")))
            q8_long_pass.append(bool((q8.get("long_context") or {}).get("passed")))

        rounds.append({
            "round": index + 1,
            "order": order,
            "f16": f16,
            "q8_0": q8,
            "q8_vs_f16_tps_ratio": round(ratio, 4) if ratio else None,
        })

    baseline_quality = median(f16_quality_scores)
    candidate_quality = median(q8_quality_scores)
    baseline_long = all(f16_long_pass) if args.run_long_context else True
    candidate_long = all(q8_long_pass) if args.run_long_context else True
    confirmation = evaluate_context_confirmation(
        model=args.model,
        context_length=args.num_ctx,
        paired_speed_ratios=ratios,
        baseline_resident_sizes_mb=f16_sizes,
        candidate_resident_sizes_mb=q8_sizes,
        baseline_quality_score=baseline_quality,
        candidate_quality_score=candidate_quality,
        baseline_long_context_passed=baseline_long,
        candidate_long_context_passed=candidate_long,
        min_quality=args.min_quality,
        min_speed_ratio=args.min_speed_ratio,
        min_win_rate=args.min_win_rate,
        min_resident_saving_mb=args.min_resident_saving_mb,
    )
    KvCacheConfirmationStore(args.confirmation_path).save(confirmation)

    report = {
        "model": args.model,
        "benchmark_design": "paired-isolated-kv-confirmation-v1",
        "num_ctx": args.num_ctx,
        "cpu_tuning": tuning.model_dump(mode="json") if tuning else None,
        "rounds": rounds,
        "summary": {
            "paired_speed_ratio_q8_vs_f16": round(median(ratios), 4) if ratios else None,
            "q8_win_rate": round(sum(1 for value in ratios if value > 1.0) / len(ratios), 4) if ratios else 0.0,
            "f16_median_resident_mb": round(median(f16_sizes), 2) if f16_sizes else None,
            "q8_median_resident_mb": round(median(q8_sizes), 2) if q8_sizes else None,
            "resident_saving_mb": confirmation.resident_saving_mb,
            "f16_quality": baseline_quality,
            "q8_quality": candidate_quality,
        },
        "confirmation": confirmation.model_dump(mode="json"),
        "runtime_default_changed": False,
        "runtime_note": (
            "This experiment records per-context evidence only. It never changes the global Ollama KV cache profile. "
            "A context-aware server/runtime design should be implemented only after confirmation passes."
        ),
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
