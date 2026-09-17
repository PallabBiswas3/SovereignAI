from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# Allow this repository script to run directly without requiring an editable
# package installation or a manually configured PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tsdiag.benchmarks.transformer_sgah_ablation import run_sgah_tfm_ablation
from tsdiag.datasets.transformer_sgah import download_sgah


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TFM / AD-TFM / TFM-AT / AD-TFM-AT ablation on SGAH")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-scale", type=float, default=1.0)
    parser.add_argument("--fixed-shift", type=float, default=0.0)
    parser.add_argument("--no-phase-augmentation", action="store_true")
    parser.add_argument("--output-dir", default="outputs/transformer_sgah_ablation")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.download:
        paths = download_sgah(data_dir)
        print(f"SGAH files ready: {len(paths)}")

    result = run_sgah_tfm_ablation(
        data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        phase_augmentation=not args.no_phase_augmentation,
        fixed_scale=args.fixed_scale,
        fixed_shift=args.fixed_shift,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(result["ranking"], indent=2))


if __name__ == "__main__":
    main()
