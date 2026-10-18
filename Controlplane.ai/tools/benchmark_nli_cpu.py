"""Offline, synthetic, paired DeBERTa CPU benchmark. Run from the repository root.

Example: python Controlplane.ai/tools/benchmark_nli_cpu.py --repetitions 5 --output local_testing/nli_cpu.json
The same fixed pairs are used for every configuration; no private evidence is read.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def cases() -> dict[str, tuple[list[str], list[str]]]:
    result = {}
    for count in (1, 3, 5, 10, 20):
        for length in ("short", "long"):
            for sharing in ("shared", "independent"):
                for outcome in ("supported", "conflicting"):
                    premises, hypotheses = [], []
                    for index in range(count):
                        asset = "Pump-102" if sharing == "shared" else f"Pump-{100 + index}"
                        claim = f"{asset} discharge pressure is 3.2 bar."
                        context = "The reading was taken at 80 percent load on 2026-09-15. "
                        if length == "long":
                            context += ("The operator recorded stable flow under normal conditions. " * 16)
                        premise = f"{asset} discharge pressure is 3.2 bar. " + context
                        premises.append(premise)
                        hypotheses.append(claim)
                        if outcome == "conflicting":
                            premises.append(f"{asset} discharge pressure is 5.4 bar. " + context)
                            hypotheses.append(claim)
                    result[f"{count}_{length}_{sharing}_{outcome}"] = (premises, hypotheses)
    return result


def percentiles(samples: list[float]) -> dict:
    ordered = sorted(samples)
    def at(p: float) -> float:
        position = (len(ordered) - 1) * p
        lower = int(position)
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction
    return {"n": len(samples), "p50_ms": at(.5), "p95_ms": at(.95),
            "p99_ms": at(.99) if len(samples) >= 100 else None,
            "mean_ms": statistics.mean(samples), "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--case", action="append", help="Exact case name; may be repeated")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    if not 1 <= args.batch_size <= 64 or (args.cpu_threads is not None and args.cpu_threads < 1):
        parser.error("batch-size must be 1..64 and cpu-threads must be positive")
    import torch
    from adaptivefact.verification.nli import TransformersNLIScorer
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    try:
        import psutil
        memory = lambda: psutil.Process().memory_info().rss
    except ImportError:
        memory = lambda: None
    model = "cross-encoder/nli-deberta-v3-small"
    started = perf_counter()
    scorer = TransformersNLIScorer(model, device="cpu", local_files_only=True, batch_size=args.batch_size)
    load_ms = (perf_counter() - started) * 1000
    workloads = cases()
    if args.case:
        unknown = set(args.case) - workloads.keys()
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        workloads = {name: workloads[name] for name in args.case}
    scorer.score(["Pump-102 pressure is 3.2 bar."], ["Pump-102 pressure is 3.2 bar."])
    samples = {name: [] for name in workloads}
    scores = {}
    for repetition in range(args.repetitions):
        names = list(workloads)
        random.Random(271828 + repetition).shuffle(names)
        for name in names:
            premises, hypotheses = workloads[name]
            started = perf_counter()
            result = scorer.score(premises, hypotheses)
            samples[name].append((perf_counter() - started) * 1000)
            if name not in scores:
                scores[name] = [[value.entailment, value.contradiction, value.neutral] for value in result]
    output = {
        "model": model, "runtime": f"torch {torch.__version__}", "device": "cpu",
        "cpu_threads": torch.get_num_threads(), "batch_size": args.batch_size,
        "load_ms": load_ms, "rss_bytes": memory(),
        "cases": {name: {"pairs": len(workloads[name][0]), "unique_pairs": len(set(zip(*workloads[name]))),
                          "latency": percentiles(values), "samples_ms": values, "scores": scores[name]}
                  for name, values in samples.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
