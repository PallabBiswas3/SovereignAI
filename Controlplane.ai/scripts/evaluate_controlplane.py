#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlplane import ControlPlane
from controlplane.evaluation import evaluate_scenarios, load_scenarios
from controlplane.verification import build_adaptive_verification_from_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ControlPlane.ai on controlled cross-risk scenarios")
    parser.add_argument("--scenarios", default="data/controlplane/scenarios.jsonl")
    parser.add_argument("--output", default="results/controlplane/evaluation.json")
    args = parser.parse_args()

    checker = ControlPlane(
        audit_enabled=False,
        verification_service=build_adaptive_verification_from_env(),
    )
    results = evaluate_scenarios(checker, load_scenarios(args.scenarios))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in results.items() if key != "rows"}, indent=2))
    print(f"\nDetailed results written to {output}")


if __name__ == "__main__":
    main()
