"""Offline INT8 candidate study; never changes the production backend.

Run from the repository root. Output is synthetic scores and timing only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from benchmark_nli_cpu import cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    import torch
    from adaptivefact.verification.nli import TransformersNLIScorer
    torch.set_num_threads(args.cpu_threads)
    scorer = TransformersNLIScorer(device="cpu", local_files_only=True)
    names = ("5_short_independent_supported", "5_short_independent_conflicting",
             "5_long_independent_supported", "5_long_independent_conflicting")
    workloads = cases()
    baseline = {}
    for name in names:
        premises, hypotheses = workloads[name]
        scorer.score(premises, hypotheses)
        started = perf_counter()
        scores = scorer.score(premises, hypotheses)
        baseline[name] = {"latency_ms": (perf_counter() - started) * 1000,
                          "scores": [[s.entailment, s.contradiction, s.neutral] for s in scores]}
    started = perf_counter()
    scorer.model = torch.ao.quantization.quantize_dynamic(
        scorer.model, {torch.nn.Linear}, dtype=torch.qint8,
    )
    quantization_ms = (perf_counter() - started) * 1000
    quantized = {}
    for name in names:
        premises, hypotheses = workloads[name]
        scorer.score(premises, hypotheses)
        started = perf_counter()
        scores = scorer.score(premises, hypotheses)
        quantized[name] = {"latency_ms": (perf_counter() - started) * 1000,
                           "scores": [[s.entailment, s.contradiction, s.neutral] for s in scores]}
    max_delta = 0.0
    class_disagreements = 0
    threshold_crossings = 0
    for name in names:
        for original, candidate in zip(baseline[name]["scores"], quantized[name]["scores"]):
            max_delta = max(max_delta, *(abs(a - b) for a, b in zip(original, candidate)))
            class_disagreements += original.index(max(original)) != candidate.index(max(candidate))
            threshold_crossings += any((a >= .72) != (b >= .72) for a, b in zip(original[:2], candidate[:2]))
    output = {"backend": "experimental PyTorch dynamic INT8 Linear", "enabled_in_production": False,
              "cpu_threads": args.cpu_threads, "quantization_ms": quantization_ms,
              "max_score_delta": max_delta, "class_disagreements": class_disagreements,
              "threshold_crossings": threshold_crossings,
              "baseline": baseline, "int8": quantized,
              "warning": "Synthetic pair study only; final ControlPlane actions were not validated."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
