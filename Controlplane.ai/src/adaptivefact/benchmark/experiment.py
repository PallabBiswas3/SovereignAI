"""
Experiment configuration and driver.

Ties together: a dataset loader, a set of components to compare, and
result persistence. This is intentionally thin in Phase 0 — no actual
risk models or verifiers exist yet — but the shape is fixed now so
later phases (risk model V1, adaptive router, ...) just add components
and configs rather than rebuilding the harness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from adaptivefact.benchmark.runner import BenchmarkRunner, Component, format_comparison_table
from adaptivefact.data.loaders.base import BaseDatasetLoader


@dataclass
class ExperimentConfig:
    name: str
    split: str | None = "test"
    max_records: int | None = None  # cap for quick smoke runs
    output_dir: str = "results"
    tags: dict[str, str] = field(default_factory=dict)


def run_experiment(
    config: ExperimentConfig,
    loader: BaseDatasetLoader,
    components: dict[str, Component],
) -> dict:
    """Loads data via `loader`, runs every component in `components`
    through the BenchmarkRunner, prints a comparison table, and writes a
    JSON result file to `config.output_dir/{name}_{timestamp}.json`.

    Returns the result dict that was written, for programmatic use
    (e.g. from a notebook or a test).
    """
    records = loader.load_list(split=config.split)
    if config.max_records is not None:
        records = records[: config.max_records]

    runner = BenchmarkRunner(components)
    reports = runner.run(records)

    print(f"Experiment: {config.name}  (dataset={loader.name}, split={config.split}, n={len(records)})")
    print(format_comparison_table(reports))

    result = {
        "experiment_name": config.name,
        "dataset": loader.name,
        "split": config.split,
        "n_records": len(records),
        "tags": config.tags,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "components": {
            key: {
                "label": report.label,
                "metrics": report.metrics,
                "latency": report.latency,
                "total_llm_calls": report.total_llm_calls,
                "total_search_calls": report.total_search_calls,
                "total_retrieval_calls": report.total_retrieval_calls,
                "total_cost": report.total_cost,
                "n": report.n,
                "skipped": report.skipped,
            }
            for key, report in reports.items()
        },
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"{config.name}_{timestamp}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults written to {out_path}")

    return result
