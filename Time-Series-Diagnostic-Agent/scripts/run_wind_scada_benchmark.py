#!/usr/bin/env python3
"""Run the complete CARE-to-Compare v6 Wind-SCADA benchmark.

This is the canonical real-data entrypoint and accepts either the extracted CARE
folder or official ZIP archive. Inference is executed through the public
``diagnose(DiagnosticRequest(...))`` boundary; labels enter only during scoring.

Usage:
    python scripts/run_wind_scada_benchmark.py --data-dir CARE_To_Compare.zip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tsdiag.benchmarks.wind_scada_public import run_care_benchmark
from tsdiag.detectors.residual_changepoint import DEFAULT_CUSUM_HOLD_SAMPLES


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full CARE v6 Wind-SCADA evaluation")
    parser.add_argument("--data-dir", required=True, help="CARE v6 directory or CARE_To_Compare.zip")
    parser.add_argument("--output-dir", default="outputs/wind_benchmark")
    parser.add_argument("--n-regimes", type=int, default=4)
    parser.add_argument("--residual-threshold", type=float)
    parser.add_argument("--persistence", type=int)
    parser.add_argument("--cusum-drift", type=float)
    parser.add_argument("--cusum-threshold", type=float)
    parser.add_argument("--cusum-hold-samples", type=int, default=DEFAULT_CUSUM_HOLD_SAMPLES)
    parser.add_argument("--criticality-threshold", type=int, default=72)
    args = parser.parse_args()

    result = run_care_benchmark(
        args.data_dir,
        n_regimes=args.n_regimes,
        residual_threshold=args.residual_threshold,
        persistence=args.persistence,
        cusum_drift=args.cusum_drift,
        cusum_threshold=args.cusum_threshold,
        cusum_hold_samples=args.cusum_hold_samples,
        criticality_threshold=args.criticality_threshold,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], indent=2))
    if result["failures"]:
        print(json.dumps({"failures": result["failures"]}, indent=2))


if __name__ == "__main__":
    main()
