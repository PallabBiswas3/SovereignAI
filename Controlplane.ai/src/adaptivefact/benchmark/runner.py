"""
Benchmark runner.

Per the discussion in chat: before building any fancy modules, every
component (risk-only, NLI, LLM judge, full agent, ...) should expose a
uniform (prediction, confidence, latency_ms, cost) interface so they can
all be dropped into the same comparison table from day one:

                 F1     Recall    P95     Calls
Risk only       0.xx    0.xx      12ms     0
NLI             0.xx    0.xx      70ms     0
LLM judge       0.xx    0.xx     800ms     1

This module defines that interface (`Component`) and the runner that
produces the table. It doesn't implement any actual components — those
land in later phases (risk/, verification/, agents/). Phase 0 only needs
the harness to exist and be provably correct, which is what
tests/test_ragtruth_loader.py checks with two trivial baseline
components.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from adaptivefact.benchmark.metrics import classification_metrics, latency_percentiles
from adaptivefact.benchmark.profiler import Timer
from adaptivefact.data.schema import ResponseRecord


@dataclass
class ComponentResult:
    """What every component must return for a single record."""

    prediction: int  # 1 = hallucinated, 0 = supported/clean
    confidence: float | None  # predicted P(hallucinated), for AUROC/calibration
    latency_ms: float | None = None  # optional internal timing; benchmark uses outer wall clock
    cost: float = 0.0  # in whatever unit the caller uses consistently
    llm_calls: int = 0
    search_calls: int = 0
    retrieval_calls: int = 0
    skip: bool = False
    skip_reason: str | None = None


class Component(abc.ABC):
    """Base class for anything the benchmark harness can compare:
    a risk-only classifier, an NLI verifier, an LLM judge, the full
    adaptive pipeline, etc."""

    #: short label used as the row name in comparison tables
    label: str

    @abc.abstractmethod
    def run(self, record: ResponseRecord) -> ComponentResult:
        raise NotImplementedError


@dataclass
class ComponentReport:
    label: str
    metrics: dict[str, float] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    total_llm_calls: int = 0
    total_search_calls: int = 0
    total_retrieval_calls: int = 0
    total_cost: float = 0.0
    n: int = 0
    skipped: int = 0


class BenchmarkRunner:
    """Runs a set of components over a set of records that have
    ground_truth_label set, and produces per-component reports."""

    def __init__(self, components: dict[str, Component]):
        self.components = components

    def run(self, records: list[ResponseRecord]) -> dict[str, ComponentReport]:
        reports: dict[str, ComponentReport] = {}

        for name, component in self.components.items():
            y_true: list[int] = []
            y_pred: list[int] = []
            y_score: list[float] = []
            latencies: list[float] = []
            costs: list[float] = []
            llm_calls = search_calls = retrieval_calls = 0
            skipped = 0

            for record in records:
                with Timer() as t:
                    result = component.run(record)

                if result.skip:
                    skipped += 1
                    continue

                truth = (
                    1
                    if record.ground_truth_label is not None
                    and record.ground_truth_label.value == "hallucinated"
                    else 0
                )
                y_true.append(truth)

                # Benchmark latency is always the outer wall-clock measurement.
                # A component may report internal timing for diagnostics, but it
                # must not replace the end-to-end time seen by the caller.
                latency = t.elapsed_ms

                y_pred.append(result.prediction)
                y_score.append(result.confidence if result.confidence is not None else float(result.prediction))
                latencies.append(latency)
                costs.append(result.cost)
                llm_calls += result.llm_calls
                search_calls += result.search_calls
                retrieval_calls += result.retrieval_calls

            reports[name] = ComponentReport(
                label=component.label,
                metrics=classification_metrics(y_true, y_pred, y_score),
                latency=latency_percentiles(latencies),
                total_llm_calls=llm_calls,
                total_search_calls=search_calls,
                total_retrieval_calls=retrieval_calls,
                total_cost=sum(costs),
                n=len(y_true),
                skipped=skipped,
            )

        return reports



def format_comparison_table(reports: dict[str, ComponentReport]) -> str:
    """Renders reports as the plain-text comparison table used
    throughout this project's planning discussion."""
    headers = ["Component", "F1", "Recall", "P95 (ms)", "LLM calls", "Search calls"]
    rows = []
    for report in reports.values():
        rows.append([
            report.label,
            f"{report.metrics.get('f1', float('nan')):.3f}",
            f"{report.metrics.get('recall', float('nan')):.3f}",
            f"{report.latency.get('p95', float('nan')):.1f}",
            str(report.total_llm_calls),
            str(report.total_search_calls),
        ])

    col_widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(w) for cell, w in zip(cells, col_widths))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in col_widths])]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)
