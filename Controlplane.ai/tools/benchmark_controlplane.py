"""Synthetic, offline ControlPlane verification benchmark.

Run separately against a clean baseline worktree and the candidate source tree.
No production document or private fixture is loaded. Each result retains only
timings, counts, statuses, and release actions.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import statistics
import sys
from pathlib import Path


def fixture(count: int, *, long: bool, shared: bool, conflicting: bool):
    from controlplane.schema import GroundingEvidence, Interaction

    claims = []
    evidence = []
    for index in range(count):
        asset = "Pump-102" if shared else f"Pump-{100 + index}"
        evidence_id = f"source-{index:03d}"
        claims.append(f"{asset} discharge pressure is 3.2 bar [{evidence_id}].")
        suffix = (" The reading was taken at 80 percent load on 2026-09-15." * 12) if long else ""
        evidence.append(GroundingEvidence(
            evidence_id=evidence_id,
            text=f"{asset} discharge pressure is 3.2 bar.{suffix}",
            document_id=f"document-{index:03d}", chunk_id=f"chunk-{index:03d}",
            revision="2", authorization_scope="synthetic-tenant",
            retrieval_score=.9, metadata={"document_status": "current"},
        ))
        if conflicting:
            evidence.append(GroundingEvidence(
                evidence_id=f"conflict-{index:03d}",
                text=f"{asset} discharge pressure is 5.4 bar.{suffix}",
                document_id=f"conflict-document-{index:03d}", chunk_id=f"conflict-chunk-{index:03d}",
                revision="2", authorization_scope="synthetic-tenant",
                retrieval_score=.9, metadata={"document_status": "current"},
            ))
    return Interaction(
        id="synthetic-verification", profile="internal_assistant",
        prompt="Summarize the recorded pump pressures.", response="\n".join(claims),
        grounding_evidence=evidence, consequential=True,
    )


def summary(samples: list[float]) -> dict:
    values = sorted(samples)
    def percentile(p):
        position = (len(values) - 1) * p
        lo = int(position)
        return values[lo] * (1 - position + lo) + values[min(lo + 1, len(values) - 1)] * (position - lo)
    return {"n": len(values), "p50_ms": percentile(.5), "p95_ms": percentile(.95),
            "p99_ms": percentile(.99) if len(values) >= 100 else None,
            "stdev_ms": statistics.stdev(values) if len(values) > 1 else None}


def cpu_worker(stop):
    value = 1
    while not stop.is_set():
        value = (value * 1664525 + 1013904223) & 0xFFFFFFFF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--case", action="append", help="Exact case name; may be repeated")
    parser.add_argument("--cpu-contention", type=int, default=0, help="Number of local CPU workers")
    parser.add_argument("--cold-case", help="Measure a first checker call before preloading (fresh process, cached weights)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    if args.cpu_contention < 0:
        parser.error("cpu-contention must be nonnegative")
    source = args.source.resolve()
    sys.path.insert(0, str(source / "src"))

    from controlplane.checker import ControlPlane
    from controlplane.policy import PolicyRepository
    from controlplane.verification import build_adaptive_verification_from_env

    service = build_adaptive_verification_from_env()
    checker = ControlPlane(policy_repository=PolicyRepository(source / "configs" / "policies"),
                           verification_service=service, audit_enabled=False)
    cases = {
        f"{count}_{'long' if long else 'short'}_{'shared' if shared else 'independent'}_{'conflicting' if conflicting else 'supported'}":
        fixture(count, long=long, shared=shared, conflicting=conflicting)
        for count in (1, 3, 5, 10, 20)
        for long in (False, True)
        for shared in (False, True)
        for conflicting in (False, True)
    }
    if args.case:
        unknown = set(args.case) - cases.keys()
        if unknown:
            parser.error(f"Unknown case names: {sorted(unknown)}")
        cases = {name: cases[name] for name in args.case}
    results = {name: [] for name in cases}
    cold_result = None
    load_ms = None
    stop = multiprocessing.Event()
    workers = [multiprocessing.Process(target=cpu_worker, args=(stop,)) for _ in range(args.cpu_contention)]
    try:
        for worker in workers:
            worker.start()
        if args.cold_case:
            if args.cold_case not in cases:
                parser.error("cold-case must be in the selected cases")
            cold = checker.check(cases[args.cold_case])
            cold_result = {"case": args.cold_case, "latency_ms": cold.total_latency_ms,
                           "action": cold.decision.action.value,
                           "budget_exceeded": cold.decision.latency_budget_exceeded}
        else:
            cold_start = __import__("time").perf_counter()
            service.phase6.nli.preload()
            load_ms = (__import__("time").perf_counter() - cold_start) * 1000
        for repetition in range(args.repetitions):
            names = list(cases)
            random.Random(314159 + repetition).shuffle(names)
            for name in names:
                report = checker.check(cases[name])
                verification = next((item for item in report.detector_results if item.detector == "adaptivefact"), None)
                metadata = verification.metadata if verification else {}
                results[name].append({
                    "latency_ms": report.total_latency_ms,
                    "action": report.decision.action.value,
                    "budget_exceeded": report.decision.latency_budget_exceeded,
                    "statuses": metadata.get("status_counts", {}),
                    "claims_checked": metadata.get("claims_checked"),
                    "nli_pairs": metadata.get("phase6", {}).get("nli_pairs"),
                    "nli_claims": metadata.get("phase6", {}).get("nli_claims"),
                    "nli_computation": metadata.get("nli_computation", {}),
                    "profile": report.profiling if hasattr(report, "profiling") else None,
                    "error": verification.error if verification else "verification_missing",
                })
    finally:
        stop.set()
        for worker in workers:
            worker.join()
    output = {"fixture": "sanitized synthetic pump facts", "source": str(source),
              "preload_including_first_forward_ms": load_ms, "cold_checker": cold_result,
              "cpu_contention_workers": args.cpu_contention,
              "cases": {name: {"latency": summary([item["latency_ms"] for item in runs]),
                               "runs": runs} for name, runs in results.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
