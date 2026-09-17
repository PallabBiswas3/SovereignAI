from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tsdiag import diagnose
from tsdiag.benchmarks.battery_nasa import run_nasa_battery_benchmark
from tsdiag.contracts import DiagnosticRequest, RunContext
from tsdiag.datasets.battery_nasa import download_nasa_battery_subset


class PublicBatteryPrognosticPipeline:
    def __init__(self, *, nominal_capacity_ah=2.0, eol_capacity_ah=1.4, minimum_observations=20, slope_window=20):
        self.config = {
            "nominal_capacity_ah": float(nominal_capacity_ah),
            "eol_capacity_ah": float(eol_capacity_ah),
            "minimum_observations": int(minimum_observations),
            "slope_window": int(slope_window),
        }

    def run(self, cycle_index, capacity_ah, *, battery_id="cell"):
        return diagnose(DiagnosticRequest(
            domain="battery",
            task="prognosis",
            policy_ref="capacity-prognosis-policy-v1",
            run_context=RunContext(
                source="NASA Ames Prognostics Center of Excellence Battery Aging Dataset",
                dataset_id="nasa-b0005-b0006-b0007-b0018",
                protocol_id="battery-capacity-cutpoint-v2",
            ),
            inputs={"cycle_index": cycle_index, "capacity_ah": capacity_ah, "battery_id": battery_id, **self.config},
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NASA battery benchmark through public diagnose boundary")
    parser.add_argument("--data-dir", default="data/nasa_battery")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    download_nasa_battery_subset(data_dir)

    payload = run_nasa_battery_benchmark(data_dir, pipeline_factory=PublicBatteryPrognosticPipeline)
    (out / "battery_nasa_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = payload["summary"]
    point = summary["point_error_metrics"]
    censored = summary["censoring_metrics"]
    print(json.dumps(summary, indent=2))
    if payload["failures"]:
        raise SystemExit(f"Benchmark failures: {payload['failures']}")
    if point["covered_case_count"] == 0:
        raise SystemExit("No exact-EOL predictions were produced.")
    if censored["covered_case_count"] == 0:
        raise SystemExit("No right-censored predictions were produced.")

    cases = payload["cases"]
    if cases:
        with (out / "battery_nasa_cases.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=cases[0].keys())
            writer.writeheader()
            writer.writerows(cases)


if __name__ == "__main__":
    main()
