from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsdiag.benchmarks.bearing_hybrid import run_lenze_drive_validation, run_paderborn_hybrid_benchmark
from tsdiag.datasets.bearing_lenze import download_lenze_mb


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bearing CNN/physics fusion benchmarks")
    parser.add_argument("--dataset", choices=("paderborn", "lenze"), required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--download", action="store_true", help="download Lenze-MB (approximately 3.2 GB)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--window-samples", type=int, default=8192)
    parser.add_argument("--waveform-samples", type=int, default=4096)
    parser.add_argument("--max-windows-per-record", type=int, default=3)
    parser.add_argument("--include-combined", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    data_dir = Path(args.data_dir or f"data/bearing_{args.dataset}")
    output_dir = Path(args.output_dir or f"outputs/bearing_{args.dataset}")
    common = {
        "window_samples": args.window_samples,
        "waveform_samples": args.waveform_samples,
        "max_windows_per_record": args.max_windows_per_record,
        "epochs": args.epochs,
        "seed": args.seed,
        "output_dir": output_dir,
    }
    if args.dataset == "lenze":
        if args.download:
            ready = download_lenze_mb(data_dir)
            print(json.dumps({key: str(value) for key, value in ready.items()}, indent=2))
        result = run_lenze_drive_validation(data_dir, **common)
    else:
        if args.download:
            parser.error("Paderborn must be downloaded from its official licensed DataCenter page")
        result = run_paderborn_hybrid_benchmark(data_dir, include_combined=args.include_combined, **common)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
