from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tsdiag.benchmarks.tep_cva import run_tep_cva_benchmark
from tsdiag.datasets.tep import download_braatz_tep
from tsdiag.tools import CVAFaultClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PCA/DPCA/CVA detection and CVA fault classification on TEP")
    parser.add_argument("--data-dir", default="data/tep_braatz")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--faults", default="0-20", help="0-20 or comma-separated IDs")
    parser.add_argument("--classifier", choices=("fda", "svm"), default="svm")
    parser.add_argument("--past-lags", type=int, default=2)
    parser.add_argument("--variance-target", type=float, default=0.95)
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--model-input", default=None, help="load a trusted persisted CVA classifier")
    parser.add_argument("--model-output", default=None, help="save the fitted classifier artifact")
    parser.add_argument("--minimum-confidence", type=float, default=0.12)
    parser.add_argument("--minimum-margin", type=float, default=0.01)
    parser.add_argument("--weak-fault-minimum-confidence", type=float, default=0.20)
    parser.add_argument("--weak-fault-minimum-margin", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/tep_cva/tep_cva_benchmark.json")
    args = parser.parse_args()

    if "-" in args.faults and "," not in args.faults:
        start, stop = (int(v) for v in args.faults.split("-", 1))
        fault_ids = list(range(start, stop + 1))
    else:
        fault_ids = [int(v.strip()) for v in args.faults.split(",") if v.strip()]
    data_dir = Path(args.data_dir)
    if args.download:
        download_braatz_tep(data_dir, fault_ids=fault_ids, splits=("train", "test"))

    fitted = None if args.model_input is None else CVAFaultClassifier.load(args.model_input)
    model_output = args.model_output
    if model_output is None and fitted is None:
        model_output = f"outputs/tep_cva/tep_cva_{args.classifier}.joblib"
    result = run_tep_cva_benchmark(
        data_dir,
        output=args.output,
        fault_ids=fault_ids,
        classifier=args.classifier,
        past_lags=args.past_lags,
        variance_target=args.variance_target,
        target_false_alarm_rate=args.target_far,
        max_samples_per_class=args.max_samples_per_class,
        seed=args.seed,
        fitted_classifier=fitted,
        model_output=model_output,
        minimum_classifier_confidence=args.minimum_confidence,
        minimum_classifier_margin=args.minimum_margin,
        weak_fault_minimum_confidence=args.weak_fault_minimum_confidence,
        weak_fault_minimum_margin=args.weak_fault_minimum_margin,
    )
    print(json.dumps({
        "detection": result["detection"]["summary"],
        "diagnosis": {k: result["diagnosis"][k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "selective")},
        "classifier_fit_seconds": result["classifier_fit_seconds"],
        "model_artifact": result["model_artifact"],
        "saved": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
