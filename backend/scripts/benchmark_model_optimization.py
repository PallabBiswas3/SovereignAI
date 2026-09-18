from __future__ import annotations

import argparse
import json
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
from app.resources.model_optimization import ModelOptimizationProfile, ModelOptimizationStore

MIB = 1024 * 1024

QUALITY_CASES = [
    {
        "name": "arithmetic",
        "prompt": "Return only the integer result of 17 * 19.",
        "contains": ["323"],
    },
    {
        "name": "industrial_extraction",
        "prompt": (
            "Read this maintenance note: Pump P-102 has RMS vibration 8.2 mm/s, "
            "bearing temperature 71 C, and discharge pressure 4.6 bar. "
            "Return only the vibration value with its unit."
        ),
        "contains": ["8.2", "mm/s"],
    },
    {
        "name": "instruction_following",
        "prompt": "Output exactly this text and nothing else: SAFE_LOCAL_ONLY",
        "exact": "SAFE_LOCAL_ONLY",
    },
    {
        "name": "numeric_comparison",
        "prompt": "Which number is larger, 0.875 or 0.857? Return only the larger number.",
        "contains": ["0.875"],
    },
    {
        "name": "ordered_extraction",
        "prompt": (
            "Given: asset=Pump-102; section=7.4; limit=7.1 mm/s. "
            "Return only: Pump-102 | 7.4 | 7.1 mm/s"
        ),
        "contains": ["Pump-102", "7.4", "7.1", "mm/s"],
    },
]


def ns_seconds(value: object) -> float | None:
    return float(value) / 1_000_000_000 if isinstance(value, (int, float)) else None


def memory_snapshot() -> dict[str, float]:
    mem = psutil.virtual_memory()
    return {
        "available_mb": round(mem.available / MIB, 2),
        "used_mb": round(mem.used / MIB, 2),
        "percent": round(float(mem.percent), 2),
    }


def local_models(client: httpx.Client, endpoint: str) -> list[dict[str, Any]]:
    response = client.get(f"{endpoint}/api/tags")
    response.raise_for_status()
    models = response.json().get("models", [])
    return [item for item in models if isinstance(item, dict)] if isinstance(models, list) else []


def resident_model(client: httpx.Client, endpoint: str, model: str) -> dict[str, Any] | None:
    response = client.get(f"{endpoint}/api/ps")
    response.raise_for_status()
    names = {model, f"{model}:latest"} if ":" not in model else {model}
    for item in response.json().get("models", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "")
        if name in names:
            return item
    return None


def unload(client: httpx.Client, endpoint: str, model: str) -> None:
    response = client.post(
        f"{endpoint}/api/generate",
        json={"model": model, "stream": False, "keep_alive": 0},
    )
    response.raise_for_status()


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
    keep_alive: str,
    temperature: float = 0.0,
) -> dict[str, Any]:
    options: dict[str, int | float] = {
        "temperature": temperature,
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
            "keep_alive": keep_alive,
            "options": options,
        },
    )
    wall = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()
    eval_seconds = ns_seconds(data.get("eval_duration"))
    eval_count = int(data.get("eval_count", 0) or 0)
    return {
        "response": str(data.get("response", "")).strip(),
        "wall_seconds": round(wall, 6),
        "load_seconds": round(ns_seconds(data.get("load_duration")) or 0.0, 6),
        "prompt_eval_seconds": round(ns_seconds(data.get("prompt_eval_duration")) or 0.0, 6),
        "decode_seconds": round(eval_seconds or 0.0, 6),
        "generated_tokens": eval_count,
        "tokens_per_second": round(eval_count / eval_seconds, 3) if eval_seconds and eval_count else None,
        "done_reason": str(data.get("done_reason") or "stop"),
    }


def quality_pass(case: dict[str, Any], response: str) -> bool:
    normalized = " ".join(response.strip().split())
    if "exact" in case:
        return normalized == str(case["exact"])
    required = [str(value).lower() for value in case.get("contains", [])]
    lowered = normalized.lower()
    return all(value in lowered for value in required)


def candidate_quantization(metadata: dict[str, Any]) -> str | None:
    details = metadata.get("details")
    if isinstance(details, dict):
        value = details.get("quantization_level")
        return str(value) if value else None
    return None


def select_candidate_names(
    base_model: str,
    model_metadata: list[dict[str, Any]],
    requested: str | None,
) -> list[str]:
    available = {
        str(item.get("name") or item.get("model") or ""): item
        for item in model_metadata
        if str(item.get("name") or item.get("model") or "")
    }
    names = [base_model]
    if requested:
        for item in requested.split(","):
            value = item.strip()
            if value and value in available and value not in names:
                names.append(value)
        return names

    # Auto-discover locally installed sibling aliases only. Never download or switch to a model
    # that the user has not explicitly installed on this machine.
    base_details = next(
        (
            item.get("details")
            for name, item in available.items()
            if name == base_model and isinstance(item.get("details"), dict)
        ),
        {},
    )
    base_family = str(base_details.get("family") or "") if isinstance(base_details, dict) else ""
    base_params = str(base_details.get("parameter_size") or "") if isinstance(base_details, dict) else ""
    for name, item in available.items():
        if name == base_model or "vision" in name.lower() or "vl" in name.lower():
            continue
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        if (
            base_family
            and str(details.get("family") or "") == base_family
            and base_params
            and str(details.get("parameter_size") or "") == base_params
        ):
            names.append(name)
    return names


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Benchmark context cost and locally installed quantization/model candidates, then persist the validated winner."
    )
    parser.add_argument("--base-model", default="qwen3:4b-instruct")
    parser.add_argument("--endpoint", default=settings.ollama_url)
    parser.add_argument("--candidates", default=None, help="Comma-separated locally installed Ollama model tags.")
    parser.add_argument("--contexts", default="2048,4096,8192")
    parser.add_argument("--performance-runs", type=int, default=2)
    parser.add_argument("--keep-alive", default="60s")
    parser.add_argument("--quality-threshold", type=float, default=settings.model_optimization_min_quality)
    parser.add_argument("--output", type=Path, default=Path("workspace/qwen3_model_optimization.json"))
    parser.add_argument("--no-save-profile", action="store_true")
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    contexts = sorted({max(512, int(value.strip())) for value in args.contexts.split(",") if value.strip()})
    cpu_tuning = CpuTuningStore(settings.model_cpu_tuning_path).get(args.base_model)
    num_thread = cpu_tuning.num_thread if cpu_tuning else None
    num_batch = cpu_tuning.num_batch if cpu_tuning else None

    with httpx.Client(timeout=httpx.Timeout(settings.model_generation_timeout_seconds)) as client:
        metadata = local_models(client, endpoint)
        names = select_candidate_names(args.base_model, metadata, args.candidates)
        available_names = {str(item.get("name") or item.get("model") or "") for item in metadata}
        if args.base_model not in available_names:
            raise RuntimeError(f"Base model '{args.base_model}' is not installed in Ollama")

        context_results: list[dict[str, Any]] = []
        perf_prompt = "Explain in four concise technical points why an outer-race bearing fault creates periodic vibration impulses."
        for num_ctx in contexts:
            unload(client, endpoint, args.base_model)
            time.sleep(0.5)
            before = memory_snapshot()
            # Excluded warm-up also forces the requested context allocation.
            generate(
                client,
                endpoint,
                model=args.base_model,
                prompt=perf_prompt,
                num_ctx=num_ctx,
                num_predict=64,
                num_thread=num_thread,
                num_batch=num_batch,
                keep_alive=args.keep_alive,
            )
            measured = [
                generate(
                    client,
                    endpoint,
                    model=args.base_model,
                    prompt=perf_prompt,
                    num_ctx=num_ctx,
                    num_predict=128,
                    num_thread=num_thread,
                    num_batch=num_batch,
                    keep_alive=args.keep_alive,
                )
                for _ in range(max(1, args.performance_runs))
            ]
            after = memory_snapshot()
            resident = resident_model(client, endpoint, args.base_model) or {}
            valid_tps = [float(item["tokens_per_second"]) for item in measured if item["tokens_per_second"]]
            context_results.append({
                "num_ctx": num_ctx,
                "median_tokens_per_second": round(median(valid_tps), 3) if valid_tps else None,
                "mean_wall_seconds": round(mean(float(item["wall_seconds"]) for item in measured), 6),
                "mean_prompt_eval_seconds": round(mean(float(item["prompt_eval_seconds"]) for item in measured), 6),
                "available_ram_before_mb": before["available_mb"],
                "available_ram_after_mb": after["available_mb"],
                "resident_size_mb": round(float(resident.get("size", 0)) / MIB, 2) if resident.get("size") else None,
                "resident_context_length": resident.get("context_length"),
                "runs": measured,
            })

        model_results: list[dict[str, Any]] = []
        metadata_by_name = {
            str(item.get("name") or item.get("model") or ""): item for item in metadata
        }
        for model in names:
            unload(client, endpoint, model)
            time.sleep(0.5)
            quality_runs = []
            for case in QUALITY_CASES:
                result = generate(
                    client,
                    endpoint,
                    model=model,
                    prompt=str(case["prompt"]),
                    num_ctx=2048,
                    num_predict=96,
                    num_thread=num_thread,
                    num_batch=num_batch,
                    keep_alive=args.keep_alive,
                )
                passed = quality_pass(case, result["response"])
                quality_runs.append({
                    "name": case["name"],
                    "passed": passed,
                    "response": result["response"],
                    "wall_seconds": result["wall_seconds"],
                })
            quality_score = sum(1 for item in quality_runs if item["passed"]) / len(quality_runs)

            perf_runs = [
                generate(
                    client,
                    endpoint,
                    model=model,
                    prompt=perf_prompt,
                    num_ctx=2048,
                    num_predict=128,
                    num_thread=num_thread,
                    num_batch=num_batch,
                    keep_alive=args.keep_alive,
                )
                for _ in range(max(1, args.performance_runs))
            ]
            resident = resident_model(client, endpoint, model) or {}
            resident_size_mb = round(float(resident.get("size", 0)) / MIB, 2) if resident.get("size") else None
            valid_tps = [float(item["tokens_per_second"]) for item in perf_runs if item["tokens_per_second"]]
            median_tps = round(median(valid_tps), 3) if valid_tps else None
            efficiency = (
                round(median_tps / (resident_size_mb / 1024.0), 4)
                if median_tps and resident_size_mb and resident_size_mb > 0
                else None
            )
            model_results.append({
                "model": model,
                "quantization": candidate_quantization(metadata_by_name.get(model, {})),
                "quality_score": round(quality_score, 3),
                "median_tokens_per_second": median_tps,
                "resident_size_mb": resident_size_mb,
                "efficiency_tps_per_gib": efficiency,
                "quality_runs": quality_runs,
                "performance_runs": perf_runs,
            })

    base = next(item for item in model_results if item["model"] == args.base_model)
    base_quality = float(base["quality_score"])
    base_efficiency = float(base["efficiency_tps_per_gib"] or 0.0)
    quality_floor = max(0.0, min(1.0, float(args.quality_threshold)))

    eligible = [
        item
        for item in model_results
        if float(item["quality_score"]) >= quality_floor
        and float(item["quality_score"]) >= base_quality - 0.05
    ]
    eligible.sort(
        key=lambda item: (
            -(float(item["efficiency_tps_per_gib"]) if item["efficiency_tps_per_gib"] else 0.0),
            -(float(item["median_tokens_per_second"]) if item["median_tokens_per_second"] else 0.0),
            float(item["resident_size_mb"] or 1e9),
        )
    )
    selected = eligible[0] if eligible else base

    # Do not switch away from the baseline for a negligible efficiency difference.
    selected_efficiency = float(selected["efficiency_tps_per_gib"] or 0.0)
    if selected["model"] != args.base_model and base_efficiency > 0:
        if selected_efficiency < base_efficiency * 1.03:
            selected = base

    saved_profile = None
    if not args.no_save_profile:
        saved_profile = ModelOptimizationProfile(
            base_model=args.base_model,
            selected_model=str(selected["model"]),
            quantization=selected.get("quantization"),
            quality_score=float(selected["quality_score"]),
            tokens_per_second=(
                float(selected["median_tokens_per_second"])
                if selected["median_tokens_per_second"] is not None
                else None
            ),
            resident_size_mb=(
                float(selected["resident_size_mb"]) if selected["resident_size_mb"] is not None else None
            ),
            context_length=2048,
        )
        ModelOptimizationStore(settings.model_optimization_path).save(saved_profile)

    report = {
        "base_model": args.base_model,
        "endpoint": endpoint,
        "cpu_tuning": cpu_tuning.model_dump(mode="json") if cpu_tuning else None,
        "quality_threshold": quality_floor,
        "candidate_models": names,
        "context_results": context_results,
        "model_results": model_results,
        "selected": selected,
        "saved_profile": saved_profile.model_dump(mode="json") if saved_profile else None,
        "selection_rule": (
            "quality >= threshold and within 0.05 of baseline; maximize decode-tps per resident GiB; "
            "require >=3% efficiency improvement before switching away from baseline"
        ),
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
