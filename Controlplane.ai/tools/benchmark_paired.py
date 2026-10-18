"""Balanced, fresh-process ABBA confirmation on fixed five-claim inputs.

Each child measures one cold checker call followed by warm calls. OS file caches
are not cleared. CPU contention is two local arithmetic worker processes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmark_nli_cpu import percentiles
from compare_controlplane_runs import flatten, metrics, RELEASE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = ("5_long_independent_supported", "5_long_shared_supported", "5_long_shared_conflicting")
    results = {}
    for workers in (0, 2):
        combined = {version: {"cases": {name: {"runs": []} for name in cases}}
                    for version in ("baseline", "candidate")}
        cold = {version: [] for version in combined}
        # Equal positions in run order: baseline, candidate, candidate, baseline.
        for index, version in enumerate(("baseline", "candidate", "candidate", "baseline")):
            output = args.output_dir / f"workers{workers}_{index}_{version}.json"
            source = args.baseline if version == "baseline" else args.candidate
            command = [sys.executable, str(Path(__file__).with_name("benchmark_controlplane.py")),
                       "--source", str(source), "--repetitions", "2", "--cpu-contention", str(workers),
                       "--cold-case", "5_long_shared_supported", "--output", str(output)]
            for case in cases:
                command.extend(("--case", case))
            subprocess.run(command, check=True)
            data = json.loads(output.read_text(encoding="utf-8"))
            cold[version].append(data["cold_checker"])
            for name in cases:
                combined[version]["cases"][name]["runs"].extend(data["cases"][name]["runs"])
            print(f"completed workers={workers} position={index} {version}", flush=True)
        result = {version: {"aggregate": metrics(flatten(combined[version])),
                            "cold_checker": cold[version],
                            "cases": {name: percentiles([r["latency_ms"] for r in item["runs"]])
                                      for name, item in combined[version]["cases"].items()}}
                  for version in combined}
        result["new_conflicting_case_releases"] = [
            [name, index] for name in cases if name.endswith("conflicting")
            for index, (before, after) in enumerate(zip(combined["baseline"]["cases"][name]["runs"],
                                                       combined["candidate"]["cases"][name]["runs"]))
            if before["action"] not in RELEASE and after["action"] in RELEASE]
        results[str(workers)] = result
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
