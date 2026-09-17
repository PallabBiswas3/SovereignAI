from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tsdiag.benchmarks.tep import TEPBenchmark
from tsdiag.datasets.tep import download_braatz_tep
from tsdiag.evaluation.reporting import write_json_report
from statsmodels.tools.sm_exceptions import InterpolationWarning


warnings.filterwarnings("ignore", category=InterpolationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="verbose is deprecated.*")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CVA-driven causal/root-cause evaluation on TEP faults")
    parser.add_argument("--data-dir", default="data/tep_braatz")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", default="outputs/tep_cva/tep_causal_benchmark.json")
    parser.add_argument("--artifact-dir", default="outputs/tep_cva/causal")
    parser.add_argument("--maxlag", type=int, default=1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    fault_ids = list(range(0, 21))
    if args.download:
        download_braatz_tep(data_dir, fault_ids=[0], splits=("train",))
        download_braatz_tep(data_dir, fault_ids=fault_ids, splits=("test",))
    benchmark = TEPBenchmark(methods=("cva",), lags_grid=(2,), maxlag=args.maxlag)
    result = benchmark.run(
        data_dir,
        fault_ids=fault_ids,
        diagnostic_faults=range(1, 21),
        diagnostic_method="cva",
        output_dir=args.artifact_dir,
        write_plots=False,
        run_ablation=False,
    )
    write_json_report(result, args.output)
    print(json.dumps({"summary": result.summary, "saved": args.output}, indent=2))


if __name__ == "__main__":
    main()
