from __future__ import annotations

import argparse
import json

from tsdiag import diagnose
from tsdiag.benchmarks.turbofan_cmapss import run_cmapss_benchmark
from tsdiag.contracts import DiagnosticRequest, RunContext


class PublicTurbofanPipeline:
    def run(self, signal_matrix, channel_names, cycle_index, **context):
        return diagnose(DiagnosticRequest(
            domain="turbofan",
            task="remaining_useful_life",
            policy_ref="turbofan-policy-v2",
            run_context=RunContext(
                source="NASA Ames Prognostics Center of Excellence C-MAPSS",
                dataset_id="cmapss-fd001-fd004",
                protocol_id="train-only-rul-v1",
            ),
            inputs={"signal_matrix": signal_matrix, "channel_names": channel_names, "cycle_index": cycle_index, **context},
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen C-MAPSS benchmark through public diagnose boundary")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/turbofan_cmapss")
    args = parser.parse_args()

    payload = run_cmapss_benchmark(
        args.data_dir,
        output_dir=args.output_dir,
        pipeline_factory=PublicTurbofanPipeline,
    )
    print(json.dumps(payload["summary"], indent=2))
    print(json.dumps(payload["by_subset"], indent=2))

    if payload["failures"]:
        raise SystemExit(f"Benchmark failures: {payload['failures']}")
    summary = payload["summary"]
    if summary["case_count"] != 707:
        raise SystemExit(f"Expected 707 test engines in the current NASA archive, got {summary['case_count']}")
    if summary["coverage"] is None or summary["coverage"] < 0.80:
        raise SystemExit(f"Coverage gate failed: {summary['coverage']}")
    if summary["rmse_cycles"] is None or summary["rmse_cycles"] > 75.0:
        raise SystemExit(f"RMSE gate failed: {summary['rmse_cycles']}")


if __name__ == "__main__":
    main()
