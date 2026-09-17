from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsdiag.evaluation.cross_domain import build_cross_domain_report, write_cross_domain_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine serialized DiagnosticResult JSON files into a versioned cross-domain report."
    )
    parser.add_argument("results", nargs="+", help="DiagnosticResult JSON files")
    parser.add_argument("--output-dir", default="outputs/cross_domain")
    parser.add_argument("--report-id")
    parser.add_argument("--scope", default="diagnostic-runs")
    args = parser.parse_args()

    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.results]
    report = build_cross_domain_report(rows, report_id=args.report_id, scope=args.scope)
    paths = write_cross_domain_report(report, args.output_dir)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
