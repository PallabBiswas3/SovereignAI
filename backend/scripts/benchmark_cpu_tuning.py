from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import httpx
import psutil

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings


def ns_to_seconds(value: object) -> float | None:
    return float(value) / 1_000_000_000 if isinstance(value, (int, float)) else None


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
    started = time.perf_counter()
    response = client.post(f"{endpoint}/api/generate", json=payload)
    wall = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()
    eval_seconds = ns_to_seconds(data.get("eval_duration"))
    eval_count = int(data.get("eval_count", 0) or 0)
    prompt_seconds = ns_to_seconds(data.get("prompt_eval_duration"))
    return {
        "num_thread": num_thread,
        "num_batch": num_batch,
        "wall_seconds": round(wall, 6),
        "load_seconds": round(ns_to_seconds(data.get("load_duration")) or 0.0, 6),
        "prompt_eval_seconds": round(prompt_seconds or 0.0, 6),
        "decode_seconds": round(eval_seconds or 0.0, 6),
        "generated_tokens": eval_count,
        "tokens_per_second": round(eval_count / eval_seconds, 3) if eval_seconds and eval_count else None,
        "done_reason": str(data.get("done_reason") or "stop"),
    }


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Tune Ollama CPU thread and batch settings on the local machine.")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--endpoint", default=settings.ollama_url)
    parser.add_argument("--prompt", default="Give a concise technical explanation of an outer-race bearing fault and its vibration signature.")
    parser.add_argument("--threads", default=None, help="Comma-separated thread counts. Defaults around physical/logical core counts.")
    parser.add_argument("--batches", default="64,128,256")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument("--keep-alive", default="60s")
    parser.add_argument("--output", type=Path, default=Path("workspace/qwen3_cpu_tuning.json"))
    args = parser.parse_args()

    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical
    if args.threads:
        threads = sorted({max(1, int(item.strip())) for item in args.threads.split(",") if item.strip()})
    else:
        threads = sorted({max(1, physical // 2), max(1, physical - 2), physical, logical})
    batches = sorted({max(1, int(item.strip())) for item in args.batches.split(",") if item.strip()})

    endpoint = args.endpoint.rstrip("/")
    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=httpx.Timeout(settings.model_generation_timeout_seconds)) as client:
        # Warm the model once so thread/batch comparisons do not include cold-load cost.
        warm = client.post(
            f"{endpoint}/api/generate",
            json={"model": args.model, "stream": False, "keep_alive": args.keep_alive},
        )
        warm.raise_for_status()

        for num_thread in threads:
            for num_batch in batches:
                runs = [
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
                    )
                    for _ in range(max(1, args.repeats))
                ]
                valid_tps = [float(run["tokens_per_second"]) for run in runs if run["tokens_per_second"] is not None]
                results.append({
                    "num_thread": num_thread,
                    "num_batch": num_batch,
                    "mean_tokens_per_second": round(mean(valid_tps), 3) if valid_tps else None,
                    "mean_wall_seconds": round(mean(float(run["wall_seconds"]) for run in runs), 6),
                    "mean_prompt_eval_seconds": round(mean(float(run["prompt_eval_seconds"]) for run in runs), 6),
                    "runs": runs,
                })

    ranked = sorted(
        results,
        key=lambda item: (
            -(float(item["mean_tokens_per_second"]) if item["mean_tokens_per_second"] is not None else -1.0),
            float(item["mean_wall_seconds"]),
        ),
    )
    best = ranked[0] if ranked else None
    report = {
        "model": args.model,
        "endpoint": endpoint,
        "cpu": {"physical_cores": physical, "logical_cores": logical},
        "benchmark": {
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
            "keep_alive": args.keep_alive,
            "repeats": max(1, args.repeats),
            "threads": threads,
            "batches": batches,
        },
        "best": best,
        "ranked_results": ranked,
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
