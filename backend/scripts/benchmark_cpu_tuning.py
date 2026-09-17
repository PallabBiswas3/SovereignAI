from __future__ import annotations

import argparse
import json
import random
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
from app.resources.cpu_tuning import CpuTuningProfile, CpuTuningStore


def ns_to_seconds(value: object) -> float | None:
    return float(value) / 1_000_000_000 if isinstance(value, (int, float)) else None


def cpu_observation() -> dict[str, float | int | None]:
    frequency = psutil.cpu_freq()
    memory = psutil.virtual_memory()
    return {
        "cpu_percent": round(psutil.cpu_percent(interval=None), 2),
        "current_frequency_mhz": round(float(frequency.current), 2) if frequency else None,
        "available_ram_mb": round(memory.available / 1024 / 1024, 2),
    }


def run_once(
    client: httpx.Client,
    endpoint: str,
    *,
    model: str,
    prompt: str,
    num_thread: int,
    num_batch: int,
    num_ctx: int,
    num_predict: int,
    keep_alive: str,
    round_index: int,
    sequence_index: int,
    measured: bool,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": 0.0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "num_thread": num_thread,
            "num_batch": num_batch,
        },
    }
    before = cpu_observation()
    started = time.perf_counter()
    response = client.post(f"{endpoint}/api/generate", json=payload)
    wall = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()
    after = cpu_observation()
    eval_seconds = ns_to_seconds(data.get("eval_duration"))
    eval_count = int(data.get("eval_count", 0) or 0)
    prompt_seconds = ns_to_seconds(data.get("prompt_eval_duration"))
    return {
        "round": round_index,
        "sequence_index": sequence_index,
        "measured": measured,
        "num_thread": num_thread,
        "num_batch": num_batch,
        "wall_seconds": round(wall, 6),
        "load_seconds": round(ns_to_seconds(data.get("load_duration")) or 0.0, 6),
        "prompt_eval_seconds": round(prompt_seconds or 0.0, 6),
        "decode_seconds": round(eval_seconds or 0.0, 6),
        "generated_tokens": eval_count,
        "tokens_per_second": round(eval_count / eval_seconds, 3) if eval_seconds and eval_count else None,
        "done_reason": str(data.get("done_reason") or "stop"),
        "system_before": before,
        "system_after": after,
    }


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Tune Ollama CPU thread and batch settings on the local machine.")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--endpoint", default=settings.ollama_url)
    parser.add_argument("--prompt", default="Give a concise technical explanation of an outer-race bearing fault and its vibration signature.")
    parser.add_argument("--threads", default=None, help="Comma-separated thread counts. Defaults around the physical-core optimum region.")
    parser.add_argument("--batches", default="64,128")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3, help="Measured interleaved rounds per configuration.")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument("--keep-alive", default="60s")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--cooldown-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("workspace/qwen3_cpu_tuning.json"))
    parser.add_argument("--no-save-profile", action="store_true")
    args = parser.parse_args()

    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical
    if args.threads:
        threads = sorted({max(1, int(item.strip())) for item in args.threads.split(",") if item.strip()})
    else:
        center = max(1, physical // 2)
        threads = sorted({max(1, center - 1), center, min(physical, center + 1)})
    batches = sorted({max(1, int(item.strip())) for item in args.batches.split(",") if item.strip()})
    candidates = [(num_thread, num_batch) for num_thread in threads for num_batch in batches]

    endpoint = args.endpoint.rstrip("/")
    rng = random.Random(args.seed)
    by_candidate: dict[tuple[int, int], dict[str, list[dict[str, Any]]]] = {
        candidate: {"warmup_runs": [], "runs": []} for candidate in candidates
    }
    execution_order: list[dict[str, int]] = []
    sequence_index = 0

    with httpx.Client(timeout=httpx.Timeout(settings.model_generation_timeout_seconds)) as client:
        preload = client.post(
            f"{endpoint}/api/generate",
            json={"model": args.model, "stream": False, "keep_alive": args.keep_alive},
        )
        preload.raise_for_status()

        for round_index in range(1, max(1, args.repeats) + 1):
            round_candidates = list(candidates)
            rng.shuffle(round_candidates)
            for num_thread, num_batch in round_candidates:
                sequence_index += 1
                execution_order.append({
                    "round": round_index,
                    "sequence_index": sequence_index,
                    "num_thread": num_thread,
                    "num_batch": num_batch,
                })
                bucket = by_candidate[(num_thread, num_batch)]
                for _ in range(max(0, args.warmups)):
                    bucket["warmup_runs"].append(
                        run_once(
                            client,
                            endpoint,
                            model=args.model,
                            prompt=args.prompt,
                            num_thread=num_thread,
                            num_batch=num_batch,
                            num_ctx=args.num_ctx,
                            num_predict=args.num_predict,
                            keep_alive=args.keep_alive,
                            round_index=round_index,
                            sequence_index=sequence_index,
                            measured=False,
                        )
                    )
                bucket["runs"].append(
                    run_once(
                        client,
                        endpoint,
                        model=args.model,
                        prompt=args.prompt,
                        num_thread=num_thread,
                        num_batch=num_batch,
                        num_ctx=args.num_ctx,
                        num_predict=args.num_predict,
                        keep_alive=args.keep_alive,
                        round_index=round_index,
                        sequence_index=sequence_index,
                        measured=True,
                    )
                )
                if args.cooldown_seconds > 0:
                    time.sleep(args.cooldown_seconds)

    results: list[dict[str, Any]] = []
    for num_thread, num_batch in candidates:
        bucket = by_candidate[(num_thread, num_batch)]
        measured_runs = bucket["runs"]
        valid_tps = [float(run["tokens_per_second"]) for run in measured_runs if run["tokens_per_second"] is not None]
        results.append({
            "num_thread": num_thread,
            "num_batch": num_batch,
            "median_tokens_per_second": round(median(valid_tps), 3) if valid_tps else None,
            "mean_tokens_per_second": round(mean(valid_tps), 3) if valid_tps else None,
            "min_tokens_per_second": round(min(valid_tps), 3) if valid_tps else None,
            "max_tokens_per_second": round(max(valid_tps), 3) if valid_tps else None,
            "median_wall_seconds": round(median(float(run["wall_seconds"]) for run in measured_runs), 6),
            "mean_prompt_eval_seconds": round(mean(float(run["prompt_eval_seconds"]) for run in measured_runs), 6),
            "warmup_runs": bucket["warmup_runs"],
            "runs": measured_runs,
        })

    ranked = sorted(
        results,
        key=lambda item: (
            -(float(item["median_tokens_per_second"]) if item["median_tokens_per_second"] is not None else -1.0),
            -(float(item["mean_tokens_per_second"]) if item["mean_tokens_per_second"] is not None else -1.0),
            float(item["median_wall_seconds"]),
        ),
    )
    best = ranked[0] if ranked else None
    saved_profile = None
    if best and not args.no_save_profile:
        saved_profile = CpuTuningProfile(
            model=args.model,
            num_thread=int(best["num_thread"]),
            num_batch=int(best["num_batch"]),
            num_ctx=args.num_ctx,
            median_tokens_per_second=(
                float(best["median_tokens_per_second"]) if best["median_tokens_per_second"] is not None else None
            ),
            mean_tokens_per_second=(
                float(best["mean_tokens_per_second"]) if best["mean_tokens_per_second"] is not None else None
            ),
            measured_runs=max(1, args.repeats),
            warmup_runs=max(0, args.warmups),
        )
        CpuTuningStore(settings.model_cpu_tuning_path).save(saved_profile)

    report = {
        "model": args.model,
        "endpoint": endpoint,
        "cpu": {"physical_cores": physical, "logical_cores": logical},
        "benchmark": {
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
            "keep_alive": args.keep_alive,
            "warmups_per_visit": max(0, args.warmups),
            "measured_rounds": max(1, args.repeats),
            "threads": threads,
            "batches": batches,
            "ranking_metric": "median_tokens_per_second",
            "methodology": "seeded_interleaved_rounds_v2",
            "seed": args.seed,
            "cooldown_seconds": max(0.0, args.cooldown_seconds),
        },
        "execution_order": execution_order,
        "best": best,
        "saved_profile": saved_profile.model_dump(mode="json") if saved_profile else None,
        "ranked_results": ranked,
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
