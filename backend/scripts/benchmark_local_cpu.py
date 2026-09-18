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
from app.resources.footprints import ModelFootprintStore, build_measured_profile

MIB = 1024 * 1024


def ns_to_seconds(value: object) -> float | None:
    return round(float(value) / 1_000_000_000, 6) if isinstance(value, (int, float)) else None


def ollama_process_rss_mb() -> float:
    total = 0
    for process in psutil.process_iter(["name", "cmdline", "memory_info"]):
        try:
            name = str(process.info.get("name") or "").lower()
            cmdline = " ".join(str(part) for part in (process.info.get("cmdline") or [])).lower()
            if "ollama" not in name and "ollama" not in cmdline:
                continue
            memory = process.info.get("memory_info")
            if memory is not None:
                total += int(memory.rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return round(total / MIB, 2)


def system_memory() -> dict[str, float]:
    memory = psutil.virtual_memory()
    return {
        "available_mb": round(memory.available / MIB, 2),
        "used_mb": round(memory.used / MIB, 2),
        "total_mb": round(memory.total / MIB, 2),
        "percent": round(float(memory.percent), 2),
    }


def unload_model(client: httpx.Client, endpoint: str, model: str) -> None:
    response = client.post(
        f"{endpoint}/api/generate",
        json={"model": model, "stream": False, "keep_alive": 0},
    )
    response.raise_for_status()


def running_model(client: httpx.Client, endpoint: str, model: str) -> dict[str, Any] | None:
    response = client.get(f"{endpoint}/api/ps")
    response.raise_for_status()
    models = response.json().get("models", [])
    names = {model, f"{model}:latest"} if ":" not in model else {model}
    for item in models if isinstance(models, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "")
        if name in names:
            return item
    return None


def stream_generate(
    client: httpx.Client,
    endpoint: str,
    model: str,
    prompt: str,
    *,
    keep_alive: str,
    num_ctx: int,
    num_predict: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "think": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": 0.2,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    final: dict[str, Any] = {}
    generated_chars = 0
    with client.stream("POST", f"{endpoint}/api/generate", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.strip():
                continue
            item = json.loads(line)
            token = str(item.get("response", ""))
            if token:
                first_token_at = first_token_at or time.perf_counter()
                generated_chars += len(token)
            if item.get("done"):
                final = item
                break
    wall = time.perf_counter() - started
    eval_seconds = ns_to_seconds(final.get("eval_duration"))
    eval_count = int(final.get("eval_count", 0) or 0)
    return {
        "wall_seconds": round(wall, 6),
        "ttft_seconds": round(first_token_at - started, 6) if first_token_at else None,
        "ollama_total_seconds": ns_to_seconds(final.get("total_duration")),
        "load_seconds": ns_to_seconds(final.get("load_duration")),
        "prompt_eval_seconds": ns_to_seconds(final.get("prompt_eval_duration")),
        "decode_seconds": eval_seconds,
        "prompt_tokens": int(final.get("prompt_eval_count", 0) or 0),
        "generated_tokens": eval_count,
        "tokens_per_second": round(eval_count / eval_seconds, 3) if eval_seconds and eval_count else None,
        "generated_chars": generated_chars,
        "done_reason": str(final.get("done_reason") or "stop"),
    }


def sovereign_chat(
    client: httpx.Client,
    base_url: str,
    prompt: str,
    *,
    model_override: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={"message": prompt, "model_override": model_override},
    )
    wall = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    return {
        "wall_seconds": round(wall, 6),
        "provider": body.get("provider"),
        "model": body.get("model"),
        "response_chars": len(str(body.get("response", ""))),
    }


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Benchmark cold/warm Ollama CPU inference and persist a measured RAM footprint."
    )
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--endpoint", default=settings.ollama_url)
    parser.add_argument(
        "--prompt",
        default="Explain in three concise points why bearing vibration increases during an outer-race fault.",
    )
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--keep-alive", default="15m")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--sovereign-url", default=None)
    parser.add_argument("--model-override", default="general")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    report: dict[str, Any] = {
        "model": args.model,
        "endpoint": endpoint,
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "benchmark": {
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
            "keep_alive": args.keep_alive,
        },
    }

    with httpx.Client(timeout=httpx.Timeout(settings.model_generation_timeout_seconds)) as client:
        # Establish a true cold baseline for the target model.
        unload_model(client, endpoint, args.model)
        time.sleep(max(0.0, args.settle_seconds))
        memory_before = system_memory()
        process_before = ollama_process_rss_mb()

        cold = stream_generate(
            client,
            endpoint,
            args.model,
            args.prompt,
            keep_alive=args.keep_alive,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
        )
        time.sleep(max(0.0, args.settle_seconds))
        memory_after = system_memory()
        process_after = ollama_process_rss_mb()
        resident = running_model(client, endpoint, args.model) or {}

        system_delta = max(0.0, memory_before["available_mb"] - memory_after["available_mb"])
        process_delta = max(0.0, process_after - process_before)
        reported_size = resident.get("size")
        reported_vram = resident.get("size_vram")
        size_mb = float(reported_size) / MIB if isinstance(reported_size, (int, float)) else None
        vram_mb = float(reported_vram) / MIB if isinstance(reported_vram, (int, float)) else 0.0
        cpu_size_mb = max(0.0, size_mb - vram_mb) if size_mb is not None else None
        context_length = resident.get("context_length")
        context_length = int(context_length) if isinstance(context_length, (int, float)) else None

        profile = build_measured_profile(
            args.model,
            system_ram_delta_mb=system_delta,
            ollama_process_rss_delta_mb=process_delta,
            ollama_reported_size_mb=size_mb,
            ollama_reported_cpu_size_mb=cpu_size_mb,
            context_length=context_length,
            sample_count=1,
            safety_multiplier=settings.model_footprint_safety_multiplier,
        )
        ModelFootprintStore(settings.model_footprints_path).save(profile)

        warm_runs = [
            stream_generate(
                client,
                endpoint,
                args.model,
                args.prompt,
                keep_alive=args.keep_alive,
                num_ctx=args.num_ctx,
                num_predict=args.num_predict,
            )
            for _ in range(max(1, args.warm_runs))
        ]

        report.update(
            {
                "memory_before": memory_before,
                "memory_after_cold_load": memory_after,
                "ollama_process_rss_before_mb": process_before,
                "ollama_process_rss_after_mb": process_after,
                "measured_system_ram_delta_mb": round(system_delta, 2),
                "measured_process_rss_delta_mb": round(process_delta, 2),
                "ollama_resident": resident,
                "persisted_footprint": profile.model_dump(mode="json"),
                "cold_run": cold,
                "warm_runs": warm_runs,
                "warm_summary": {
                    "mean_wall_seconds": round(mean(run["wall_seconds"] for run in warm_runs), 6),
                    "mean_ttft_seconds": round(
                        mean(run["ttft_seconds"] for run in warm_runs if run["ttft_seconds"] is not None),
                        6,
                    ),
                    "mean_tokens_per_second": round(
                        mean(
                            run["tokens_per_second"]
                            for run in warm_runs
                            if run["tokens_per_second"] is not None
                        ),
                        3,
                    ),
                },
            }
        )

        if args.sovereign_url:
            report["sovereign_chat"] = sovereign_chat(
                client,
                args.sovereign_url,
                args.prompt,
                model_override=args.model_override,
            )

    serialized = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
