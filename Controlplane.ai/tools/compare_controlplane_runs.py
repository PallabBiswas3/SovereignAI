"""Compare two sanitized benchmark JSON files without reading evidence text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_nli_cpu import percentiles


RELEASE = {"allow", "allow_with_warning", "redact"}


def flatten(report):
    return [(name, index, run) for name, case in report["cases"].items()
            for index, run in enumerate(case["runs"])]


def metrics(rows):
    runs = [run for _, _, run in rows]
    conflicts = [(name, run) for name, _, run in rows if name.endswith("_conflicting")]
    supported = [(name, run) for name, _, run in rows if name.endswith("_supported")]
    total_claims = sum(sum(run["statuses"].values()) for run in runs)
    abstained = sum(run["statuses"].get(key, 0) for run in runs
                    for key in ("unknown", "undecidable", "unverified"))
    accuracy_hits = sum(bool(run["statuses"]) and all(key == "supported" for key in run["statuses"])
                        for _, run in supported) + sum(
                            bool({"conflicting", "contradicted"} & run["statuses"].keys())
                            for _, run in conflicts)
    correct_claims = sum(run["statuses"].get("supported", 0) for _, run in supported) + sum(
        run["statuses"].get("conflicting", 0) + run["statuses"].get("contradicted", 0)
        for _, run in conflicts)
    inference_ms = sum(run["nli_computation"].get("inference_total_ms", 0.0) for run in runs)
    inferred_pairs = sum(run["nli_computation"].get("inferred_pairs", 0) for run in runs)
    return {
        "latency": percentiles([run["latency_ms"] for run in runs]),
        "synthetic_case_outcome_accuracy": accuracy_hits / len(runs),
        "synthetic_claim_outcome_accuracy": correct_claims / total_claims if total_claims else None,
        "conflicting_case_releases": sum(run["action"] in RELEASE for _, run in conflicts),
        "supported_case_holds": sum(run["action"] not in RELEASE for _, run in supported),
        "abstention_claim_fraction": abstained / total_claims if total_claims else None,
        "nli_pairs": sum(run["nli_pairs"] or 0 for run in runs),
        "nli_inferred_pairs": inferred_pairs,
        "nli_throughput_pairs_per_second": 1000 * inferred_pairs / inference_ms if inference_ms else None,
        "forward_passes": sum(run["nli_computation"].get("forward_passes", 0) for run in runs)
                          if any("forward_passes" in run["nli_computation"] for run in runs) else None,
        "tokens_processed": sum(run["nli_computation"].get("tokens_processed", 0) for run in runs)
                            if any("tokens_processed" in run["nli_computation"] for run in runs) else None,
        "peak_rss_bytes": max((run["profile"] or {}).get("process_rss_bytes") or 0 for run in runs) or None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    optimized = json.loads(args.optimized.read_text(encoding="utf-8"))
    left = {(name, index): run for name, index, run in flatten(baseline)}
    right = {(name, index): run for name, index, run in flatten(optimized)}
    if left.keys() != right.keys():
        raise ValueError("Benchmark cases/repetitions do not match")
    changed_statuses = []
    changed_actions = []
    new_conflict_releases = []
    for key in sorted(left):
        before, after = left[key], right[key]
        if before["statuses"] != after["statuses"]:
            changed_statuses.append(key)
        if before["action"] != after["action"]:
            changed_actions.append(key)
        if key[0].endswith("_conflicting") and before["action"] not in RELEASE and after["action"] in RELEASE:
            new_conflict_releases.append(key)
    stage_keys = ("policy_load", "hallucination_inclusive", "parallel_detectors_wall",
                  "adaptive_verification_inclusive", "policy_evaluation", "response_transform")
    optimized_runs = list(right.values())
    stage_metrics = {key: percentiles([(run["profile"] or {}).get("stage_ms", {}).get(key, 0.0)
                                       for run in optimized_runs]) for key in stage_keys}
    output = {
        "baseline": metrics(flatten(baseline)), "optimized": metrics(flatten(optimized)),
        "changed_statuses": changed_statuses, "changed_actions": changed_actions,
        "new_conflicting_case_releases": new_conflict_releases,
        "optimized_stages": stage_metrics,
        "limitations": "Synthetic fixture labels are workload proxies, not adjudicated claim truth. This comparison does not establish production safety.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("baseline", "optimized", "changed_statuses", "changed_actions", "new_conflicting_case_releases")}, indent=2))


if __name__ == "__main__":
    main()
