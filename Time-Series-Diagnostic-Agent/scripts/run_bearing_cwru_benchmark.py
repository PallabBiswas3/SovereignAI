from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsdiag import diagnose
from tsdiag.benchmarks.bearing_cwru import run_cwru_benchmark
from tsdiag.contracts import DiagnosticRequest, RunContext
from tsdiag.datasets.bearing_cwru import download_cwru_007_drive_end


class PublicBearingPipeline:
    def __init__(self, *, minimum_confidence: float = 0.45, minimum_harmonics: int = 2):
        self.minimum_confidence = float(minimum_confidence)
        self.minimum_harmonics = int(minimum_harmonics)

    def run(self, signal, sampling_rate_hz: float, *, fault_frequencies=None, shaft_rate_hz=None, channel_name="ch0", operating_condition=None):
        return diagnose(DiagnosticRequest(
            domain="bearing",
            task="fault_diagnosis",
            policy_ref="bearing-policy-v2",
            run_context=RunContext(
                source="CWRU Bearing Data Center",
                dataset_id="cwru-12k-drive-end-skf-007",
                protocol_id="cwru-fixed-record-v1",
            ),
            inputs={
                "signal": signal,
                "sampling_rate_hz": sampling_rate_hz,
                "fault_frequencies": dict(fault_frequencies or {}),
                "shaft_rate_hz": shaft_rate_hz,
                "channel_name": channel_name,
                "operating_condition": dict(operating_condition or {}),
                "minimum_confidence": self.minimum_confidence,
                "minimum_harmonics": self.minimum_harmonics,
            },
        ))


def _assert_locked_parity(summary: dict) -> None:
    expected = {
        "record_count": 16,
        "all_record_accuracy": 0.3125,
        "all_record_macro_f1": 0.35,
        "coverage": 0.3125,
        "abstention_rate": 0.6875,
        "covered_accuracy": 1.0,
        "normal_false_alarm_rate": 0.0,
        "normal_abstention_rate": 1.0,
        "fault_detection_rate": 0.4166666666666667,
    }
    for key, value in expected.items():
        actual = summary.get(key)
        if isinstance(value, float):
            if actual is None or abs(float(actual) - value) > 1e-12:
                raise SystemExit(f"CWRU parity gate failed for {key}: expected {value}, got {actual}")
        elif actual != value:
            raise SystemExit(f"CWRU parity gate failed for {key}: expected {value}, got {actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled CWRU 0.007-in drive-end bearing benchmark")
    parser.add_argument("--data-dir", default="data/cwru_007_de")
    parser.add_argument("--download", action="store_true", help="download the fixed CWRU manifest before evaluation")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--max-windows", type=int, default=6)
    parser.add_argument("--minimum-confidence", type=float, default=0.45)
    parser.add_argument("--minimum-harmonics", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/bearing_cwru")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.download:
        paths = download_cwru_007_drive_end(data_dir)
        print(f"CWRU files ready: {len(paths)}")

    result = run_cwru_benchmark(
        data_dir,
        window_seconds=args.window_seconds,
        max_windows=args.max_windows,
        minimum_confidence=args.minimum_confidence,
        minimum_harmonics=args.minimum_harmonics,
        output_dir=args.output_dir,
        pipeline_factory=PublicBearingPipeline,
    )
    print(json.dumps(result["summary"], indent=2))
    if result["failures"]:
        print("Failures:")
        print(json.dumps(result["failures"], indent=2))
        raise SystemExit(2)
    _assert_locked_parity(result["summary"])


if __name__ == "__main__":
    main()
