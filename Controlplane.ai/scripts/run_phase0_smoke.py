#!/usr/bin/env python3
"""
Runs the Phase 0 smoke experiment from configs/phase0_ragtruth.yaml.

This doesn't test any real system component (none exist yet) — it
proves the schema/loader/harness/logging plumbing works end-to-end on
the full real dataset, using two trivial baselines as stand-ins:
  - a "never hallucinated" floor,
  - an oracle that reads the ground-truth label directly (sanity check
    that the harness can tell a perfect predictor from a useless one).

Phase 2 replaces these with an actual risk-only classifier.

Usage:
    python scripts/run_phase0_smoke.py
    python scripts/run_phase0_smoke.py --config configs/phase0_ragtruth.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from adaptivefact.benchmark.experiment import ExperimentConfig, run_experiment
from adaptivefact.benchmark.runner import Component, ComponentResult
from adaptivefact.data.loaders.ragtruth import RAGTruthLoader
from adaptivefact.utils.logging import RequestLogger


class AlwaysSupportedBaseline(Component):
    label = "Always-supported baseline"

    def run(self, record) -> ComponentResult:
        return ComponentResult(prediction=0, confidence=0.0, latency_ms=0.01)


class OracleBaseline(Component):
    """NOT a real component — reads the label directly. Sanity-check only."""

    label = "Oracle (uses ground truth)"

    def run(self, record) -> ComponentResult:
        has_spans = len(record.ground_truth_spans) > 0
        return ComponentResult(prediction=int(has_spans), confidence=float(has_spans), latency_ms=0.01)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase0_ragtruth.yaml")
    args = parser.parse_args()

    raw_config = yaml.safe_load(Path(args.config).read_text())

    exp_cfg = raw_config["experiment"]
    ds_cfg = raw_config["dataset"]

    loader = RAGTruthLoader(
        root=ds_cfg["root"],
        include_low_quality=ds_cfg.get("include_low_quality", False),
    )

    config = ExperimentConfig(
        name=exp_cfg["name"],
        split=exp_cfg.get("split"),
        max_records=exp_cfg.get("max_records"),
        output_dir=exp_cfg.get("output_dir", "results"),
        tags={"dataset_loader": ds_cfg["loader"]},
    )

    components = {
        "always_supported": AlwaysSupportedBaseline(),
        "oracle": OracleBaseline(),
    }

    result = run_experiment(config, loader, components)

    # Also demonstrate the §57 structured request logger on a handful of
    # records, so the log format can be inspected.
    logger = RequestLogger("results/phase0/phase0_sample_requests.jsonl")
    sample_records = loader.load_list(split=config.split)[:5]
    for record in sample_records:
        logger.log_from_record(record)
    print("\nSample §57-format request logs written to results/phase0/phase0_sample_requests.jsonl")

    return result


if __name__ == "__main__":
    main()
