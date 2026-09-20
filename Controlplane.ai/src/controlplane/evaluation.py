from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from controlplane.checker import ControlPlane
from controlplane.schema import EnforcementAction, Interaction


ACTION_RANK = {
    EnforcementAction.ALLOW.value: 0,
    EnforcementAction.WARN.value: 1,
    EnforcementAction.REDACT.value: 2,
    EnforcementAction.REVIEW.value: 3,
    EnforcementAction.BLOCK.value: 4,
}


def load_scenarios(path: str | Path) -> list[dict[str, Any]]:
    scenarios = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                scenarios.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid scenario JSON on line {line_number}") from exc
    return scenarios


def evaluate_scenarios(checker: ControlPlane, scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    category_tp: dict[str, int] = {}
    category_fp: dict[str, int] = {}
    category_fn: dict[str, int] = {}
    subtype_tp: dict[str, int] = {}
    subtype_fp: dict[str, int] = {}
    subtype_fn: dict[str, int] = {}
    latencies = []

    unknown_reason_finding_counts: dict[str, int] = {}
    unknown_reason_scenario_counts: dict[str, int] = {}
    unknown_reason_tp_scenarios: dict[str, int] = {}
    unknown_reason_fp_scenarios: dict[str, int] = {}
    unknown_reason_over_intervention_scenarios: dict[str, int] = {}

    for scenario in scenarios:
        interaction = Interaction.model_validate(scenario["interaction"])
        report = checker.check(interaction)
        expected_action = EnforcementAction(scenario["expected_action"])
        expected_categories = set(scenario.get("expected_categories", []))
        predicted_categories = {finding.category.value for finding in report.findings}
        expected_subtypes = set(scenario.get("expected_subtypes", []))
        gold_statuses = {claim.get("status") for claim in scenario.get("gold_claims", [])}
        if "contradicted" in gold_statuses:
            expected_subtypes.add("claim_contradicted")
        if "unknown" in gold_statuses:
            expected_subtypes.add("claim_unknown")
        predicted_subtypes = {finding.subtype for finding in report.findings}

        for category in expected_categories | predicted_categories:
            if category in expected_categories and category in predicted_categories:
                category_tp[category] = category_tp.get(category, 0) + 1
            elif category in predicted_categories:
                category_fp[category] = category_fp.get(category, 0) + 1
            else:
                category_fn[category] = category_fn.get(category, 0) + 1

        for subtype in expected_subtypes | predicted_subtypes:
            if subtype in expected_subtypes and subtype in predicted_subtypes:
                subtype_tp[subtype] = subtype_tp.get(subtype, 0) + 1
            elif subtype in predicted_subtypes:
                subtype_fp[subtype] = subtype_fp.get(subtype, 0) + 1
            else:
                subtype_fn[subtype] = subtype_fn.get(subtype, 0) + 1

        unknown_findings = [
            finding for finding in report.findings if finding.subtype == "claim_unknown"
        ]
        unknown_reasons = {
            str(finding.metadata.get("unknown_reason") or "unspecified")
            for finding in unknown_findings
        }
        for finding in unknown_findings:
            reason = str(finding.metadata.get("unknown_reason") or "unspecified")
            unknown_reason_finding_counts[reason] = unknown_reason_finding_counts.get(reason, 0) + 1
        expected_unknown = "claim_unknown" in expected_subtypes
        for reason in unknown_reasons:
            unknown_reason_scenario_counts[reason] = unknown_reason_scenario_counts.get(reason, 0) + 1
            if expected_unknown:
                unknown_reason_tp_scenarios[reason] = unknown_reason_tp_scenarios.get(reason, 0) + 1
            else:
                unknown_reason_fp_scenarios[reason] = unknown_reason_fp_scenarios.get(reason, 0) + 1
            if (
                expected_action == EnforcementAction.ALLOW
                and ACTION_RANK[report.decision.action.value] > ACTION_RANK[EnforcementAction.ALLOW.value]
            ):
                unknown_reason_over_intervention_scenarios[reason] = (
                    unknown_reason_over_intervention_scenarios.get(reason, 0) + 1
                )

        rows.append(
            {
                "id": scenario.get("id", interaction.id),
                "profile": interaction.profile,
                "expected_action": expected_action.value,
                "predicted_action": report.decision.action.value,
                "action_correct": report.decision.action == expected_action,
                "expected_categories": sorted(expected_categories),
                "predicted_categories": sorted(predicted_categories),
                "expected_subtypes": sorted(expected_subtypes),
                "predicted_subtypes": sorted(predicted_subtypes),
                "unknown_reasons": sorted(unknown_reasons),
                "latency_ms": report.total_latency_ms,
            }
        )
        latencies.append(report.total_latency_ms)

    def classification_summary(
        true_positive: dict[str, int],
        false_positive: dict[str, int],
        false_negative: dict[str, int],
    ) -> dict[str, dict[str, float | int]]:
        summary = {}
        for label in sorted(set(true_positive) | set(false_positive) | set(false_negative)):
            tp = true_positive.get(label, 0)
            fp = false_positive.get(label, 0)
            fn = false_negative.get(label, 0)
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            summary[label] = {
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / max(1e-12, precision + recall),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
            }
        return summary

    categories = classification_summary(category_tp, category_fp, category_fn)
    subtypes = classification_summary(subtype_tp, subtype_fp, subtype_fn)

    unknown_reason_metrics: dict[str, dict[str, float | int]] = {}
    for reason in sorted(unknown_reason_scenario_counts):
        predicted = unknown_reason_scenario_counts.get(reason, 0)
        tp = unknown_reason_tp_scenarios.get(reason, 0)
        fp = unknown_reason_fp_scenarios.get(reason, 0)
        unknown_reason_metrics[reason] = {
            "findings": unknown_reason_finding_counts.get(reason, 0),
            "predicted_scenarios": predicted,
            "true_positive_scenarios": tp,
            "false_positive_scenarios": fp,
            "scenario_precision": tp / max(1, tp + fp),
            "over_intervention_scenarios": unknown_reason_over_intervention_scenarios.get(reason, 0),
        }

    unsafe_expected = [row for row in rows if row["expected_action"] != EnforcementAction.ALLOW.value]
    unsafe_allowed = [row for row in unsafe_expected if row["predicted_action"] == EnforcementAction.ALLOW.value]
    expected_action_counts: dict[str, int] = {}
    predicted_action_counts: dict[str, int] = {}
    action_confusion: dict[str, dict[str, int]] = {}
    for row in rows:
        expected = row["expected_action"]
        predicted = row["predicted_action"]
        expected_action_counts[expected] = expected_action_counts.get(expected, 0) + 1
        predicted_action_counts[predicted] = predicted_action_counts.get(predicted, 0) + 1
        action_confusion.setdefault(expected, {})[predicted] = (
            action_confusion.setdefault(expected, {}).get(predicted, 0) + 1
        )
    expected_allow = [row for row in rows if row["expected_action"] == EnforcementAction.ALLOW.value]
    over_intervened = [
        row for row in expected_allow
        if ACTION_RANK[row["predicted_action"]] > ACTION_RANK[EnforcementAction.ALLOW.value]
    ]
    arr = np.asarray(latencies, dtype=float)
    return {
        "n": len(rows),
        "action_accuracy": sum(row["action_correct"] for row in rows) / max(1, len(rows)),
        "unsafe_allow_rate": len(unsafe_allowed) / max(1, len(unsafe_expected)),
        "over_intervention_rate": len(over_intervened) / max(1, len(expected_allow)),
        "human_review_rate": predicted_action_counts.get(EnforcementAction.REVIEW.value, 0) / max(1, len(rows)),
        "expected_action_counts": expected_action_counts,
        "predicted_action_counts": predicted_action_counts,
        "action_confusion": action_confusion,
        "category_metrics": categories,
        "subtype_metrics": subtypes,
        "unknown_reason_metrics": unknown_reason_metrics,
        "latency_ms": {
            "p50": float(np.percentile(arr, 50)) if len(arr) else 0.0,
            "p95": float(np.percentile(arr, 95)) if len(arr) else 0.0,
            "p99": float(np.percentile(arr, 99)) if len(arr) else 0.0,
            "mean": float(np.mean(arr)) if len(arr) else 0.0,
            "max": float(np.max(arr)) if len(arr) else 0.0,
        },
        "rows": rows,
    }
