#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

from controlplane import ControlPlane, Interaction
from controlplane.evaluation import load_scenarios
from controlplane.verification import build_adaptive_verification_from_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent ControlPlane stress runner")
    parser.add_argument("--scenarios", default="data/controlplane/benchmark_v1/validation_candidates.jsonl")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", default="results/controlplane/stress.json")
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenarios)
    checker = ControlPlane(
        audit_enabled=False,
        verification_service=build_adaptive_verification_from_env(),
    )

    def execute(index: int) -> dict:
        scenario = scenarios[index % len(scenarios)]
        started = perf_counter()
        report = checker.check(Interaction.model_validate(scenario["interaction"]))
        return {
            "latency_ms": (perf_counter() - started) * 1000.0,
            "action": report.decision.action.value,
            "budget_exceeded": report.decision.latency_budget_exceeded,
        }

    for index in range(max(0, args.warmup)):
        execute(index)

    started = perf_counter()
    completed = []
    errors = []
    with ThreadPoolExecutor(max_workers=args.concurrency, thread_name_prefix="stress") as executor:
        futures = {executor.submit(execute, index): index for index in range(args.requests)}
        for future in as_completed(futures):
            try:
                completed.append(future.result())
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
    wall_seconds = perf_counter() - started

    latencies = np.asarray([item["latency_ms"] for item in completed], dtype=float)
    actions = Counter(item["action"] for item in completed)
    report = {
        "requested": args.requests,
        "completed": len(completed),
        "errors": len(errors),
        "error_samples": errors[:10],
        "concurrency": args.concurrency,
        "warmup_requests_excluded": max(0, args.warmup),
        "wall_seconds": wall_seconds,
        "throughput_requests_per_second": len(completed) / max(wall_seconds, 1e-12),
        "action_counts": dict(actions),
        "latency_budget_exceeded_rate": sum(item["budget_exceeded"] for item in completed) / max(1, len(completed)),
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50)) if len(latencies) else 0.0,
            "p95": float(np.percentile(latencies, 95)) if len(latencies) else 0.0,
            "p99": float(np.percentile(latencies, 99)) if len(latencies) else 0.0,
            "mean": float(np.mean(latencies)) if len(latencies) else 0.0,
            "max": float(np.max(latencies)) if len(latencies) else 0.0,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
