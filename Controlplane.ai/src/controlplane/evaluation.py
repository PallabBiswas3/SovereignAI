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

V3_UNRESOLVED_SUBTYPES = {"claim_undecidable", "claim_unsupported", "claim_conflicting"}


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


def _increment(mapping: dict[str, int], key: str, amount: int = 1) -> None:
    mapping[key] = mapping.get(key, 0) + amount


def evaluate_scenarios(checker: ControlPlane, scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    category_tp: dict[str, int] = {}
    category_fp: dict[str, int] = {}
    category_fn: dict[str, int] = {}
    subtype_tp: dict[str, int] = {}
    subtype_fp: dict[str, int] = {}
    subtype_fn: dict[str, int] = {}
    latencies: list[float] = []
    factuality_state_counts: dict[str, int] = {}
    verification_depth_counts: dict[str, int] = {}
    over_intervention_by_subtype: dict[str, int] = {}
    unresolved_tp = unresolved_fp = unresolved_fn = 0

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
        expected_unresolved = "unknown" in gold_statuses or "claim_unknown" in expected_subtypes
        if expected_unresolved:
            expected_subtypes.add("claim_unknown")

        raw_predicted_subtypes = {finding.subtype for finding in report.findings}
        predicted_subtypes_for_legacy_eval = set(raw_predicted_subtypes)
        predicted_unresolved = bool(raw_predicted_subtypes & V3_UNRESOLVED_SUBTYPES) or "claim_unknown" in raw_predicted_subtypes
        if predicted_unresolved:
            predicted_subtypes_for_legacy_eval.add("claim_unknown")

        if expected_unresolved and predicted_unresolved:
            unresolved_tp += 1
        elif predicted_unresolved:
            unresolved_fp += 1
        elif expected_unresolved:
            unresolved_fn += 1

        for result in report.detector_results:
            if result.detector != "adaptivefact":
                continue
            for state, count in (result.metadata.get("status_counts") or {}).items():
                factuality_state_counts[str(state)] = factuality_state_counts.get(str(state), 0) + int(count)

        depth = report.decision.verification_depth.value
        verification_depth_counts[depth] = verification_depth_counts.get(depth, 0) + 1

        for category in expected_categories | predicted_categories:
            if category in expected_categories and category in predicted_categories:
                category_tp[category] = category_tp.get(category, 0) + 1
            elif category in predicted_categories:
                category_fp[category] = category_fp.get(category, 0) + 1
            else:
                category_fn[category] = category_fn.get(category, 0) + 1

        for subtype in expected_subtypes | predicted_subtypes_for_legacy_eval:
            if subtype in expected_subtypes and subtype in predicted_subtypes_for_legacy_eval:
                subtype_tp[subtype] = subtype_tp.get(subtype, 0) + 1
            elif subtype in predicted_subtypes_for_legacy_eval:
                subtype_fp[subtype] = subtype_fp.get(subtype, 0) + 1
            else:
                subtype_fn[subtype] = subtype_fn.get(subtype, 0) + 1

        over_intervened = (
            expected_action == EnforcementAction.ALLOW
            and ACTION_RANK[report.decision.action.value] > ACTION_RANK[EnforcementAction.ALLOW.value]
        )
        if over_intervened:
            for subtype in raw_predicted_subtypes:
                over_intervention_by_subtype[subtype] = over_intervention_by_subtype.get(subtype, 0) + 1

        rows.append(
            {
                "id": scenario.get("id", interaction.id),
                "profile": interaction.profile,
                "consequential": bool(interaction.consequential),
                "expected_action": expected_action.value,
                "predicted_action": report.decision.action.value,
                "action_correct": report.decision.action == expected_action,
                "expected_categories": sorted(expected_categories),
                "predicted_categories": sorted(predicted_categories),
                "expected_subtypes": sorted(expected_subtypes),
                "predicted_subtypes": sorted(raw_predicted_subtypes),
                "legacy_predicted_subtypes": sorted(predicted_subtypes_for_legacy_eval),
                "expected_unresolved": expected_unresolved,
                "predicted_unresolved": predicted_unresolved,
                "verification_depth": depth,
                "latency_ms": report.total_latency_ms,
            }
        )
        latencies.append(report.total_latency_ms)

    def classification_summary(true_positive: dict[str, int], false_positive: dict[str, int], false_negative: dict[str, int]) -> dict[str, dict[str, float | int]]:
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
    unresolved_precision = unresolved_tp / max(1, unresolved_tp + unresolved_fp)
    unresolved_recall = unresolved_tp / max(1, unresolved_tp + unresolved_fn)

    unsafe_expected = [row for row in rows if row["expected_action"] != EnforcementAction.ALLOW.value]
    unsafe_allowed = [row for row in unsafe_expected if row["predicted_action"] == EnforcementAction.ALLOW.value]

    unsafe_by_expected_action: dict[str, int] = {}
    unsafe_by_expected_category: dict[str, int] = {}
    unsafe_by_expected_subtype: dict[str, int] = {}
    unsafe_by_profile: dict[str, int] = {}
    unsafe_by_verification_depth: dict[str, int] = {}
    unsafe_by_consequential: dict[str, int] = {}
    unsafe_by_predicted_subtype: dict[str, int] = {}

    for row in unsafe_allowed:
        _increment(unsafe_by_expected_action, str(row["expected_action"]))
        _increment(unsafe_by_profile, str(row["profile"]))
        _increment(unsafe_by_verification_depth, str(row["verification_depth"]))
        _increment(unsafe_by_consequential, "consequential" if row["consequential"] else "non_consequential")
        expected_category_labels = row["expected_categories"] or ["<none>"]
        for category in expected_category_labels:
            _increment(unsafe_by_expected_category, str(category))
        expected_subtype_labels = row["expected_subtypes"] or ["<none>"]
        for subtype in expected_subtype_labels:
            _increment(unsafe_by_expected_subtype, str(subtype))
        predicted_subtype_labels = row["predicted_subtypes"] or ["<none>"]
        for subtype in predicted_subtype_labels:
            _increment(unsafe_by_predicted_subtype, str(subtype))

    expected_action_counts: dict[str, int] = {}
    predicted_action_counts: dict[str, int] = {}
    action_confusion: dict[str, dict[str, int]] = {}
    for row in rows:
        expected = row["expected_action"]
        predicted = row["predicted_action"]
        expected_action_counts[expected] = expected_action_counts.get(expected, 0) + 1
        predicted_action_counts[predicted] = predicted_action_counts.get(predicted, 0) + 1
        action_confusion.setdefault(expected, {})[predicted] = action_confusion.setdefault(expected, {}).get(predicted, 0) + 1

    expected_allow = [row for row in rows if row["expected_action"] == EnforcementAction.ALLOW.value]
    over_intervened = [row for row in expected_allow if ACTION_RANK[row["predicted_action"]] > ACTION_RANK[EnforcementAction.ALLOW.value]]
    arr = np.asarray(latencies, dtype=float)
    return {
        "n": len(rows),
        "action_accuracy": sum(row["action_correct"] for row in rows) / max(1, len(rows)),
        "unsafe_allow_rate": len(unsafe_allowed) / max(1, len(unsafe_expected)),
        "unsafe_allow_count": len(unsafe_allowed),
        "over_intervention_rate": len(over_intervened) / max(1, len(expected_allow)),
        "human_review_rate": predicted_action_counts.get(EnforcementAction.REVIEW.value, 0) / max(1, len(rows)),
        "expected_action_counts": expected_action_counts,
        "predicted_action_counts": predicted_action_counts,
        "action_confusion": action_confusion,
        "unsafe_allow_breakdown": {
            "by_expected_action": dict(sorted(unsafe_by_expected_action.items(), key=lambda item: item[1], reverse=True)),
            "by_expected_category": dict(sorted(unsafe_by_expected_category.items(), key=lambda item: item[1], reverse=True)),
            "by_expected_subtype": dict(sorted(unsafe_by_expected_subtype.items(), key=lambda item: item[1], reverse=True)),
            "by_profile": dict(sorted(unsafe_by_profile.items(), key=lambda item: item[1], reverse=True)),
            "by_verification_depth": dict(sorted(unsafe_by_verification_depth.items(), key=lambda item: item[1], reverse=True)),
            "by_consequential": dict(sorted(unsafe_by_consequential.items(), key=lambda item: item[1], reverse=True)),
            "by_predicted_subtype": dict(sorted(unsafe_by_predicted_subtype.items(), key=lambda item: item[1], reverse=True)),
        },
        "unsafe_allow_examples": unsafe_allowed[:25],
        "category_metrics": categories,
        "subtype_metrics_legacy_compatible": subtypes,
        "unresolved_aggregate_metrics": {
            "precision": unresolved_precision,
            "recall": unresolved_recall,
            "f1": 2 * unresolved_precision * unresolved_recall / max(1e-12, unresolved_precision + unresolved_recall),
            "true_positive": unresolved_tp,
            "false_positive": unresolved_fp,
            "false_negative": unresolved_fn,
        },
        "factuality_state_counts": factuality_state_counts,
        "verification_depth_counts": verification_depth_counts,
        "over_intervention_by_subtype": dict(sorted(over_intervention_by_subtype.items(), key=lambda item: item[1], reverse=True)),
        "evaluation_note": (
            "The legacy synthetic corpus labels unresolved claims primarily as 'unknown'. "
            "It can evaluate unresolved-vs-resolved behavior, but it cannot establish semantic precision "
            "for the new UNSUPPORTED, UNDECIDABLE, and CONFLICTING states. Those require structured "
            "evidence and independently adjudicated state labels."
        ),
        "latency_ms": {
            "p50": float(np.percentile(arr, 50)) if len(arr) else 0.0,
            "p95": float(np.percentile(arr, 95)) if len(arr) else 0.0,
            "p99": float(np.percentile(arr, 99)) if len(arr) else 0.0,
            "mean": float(np.mean(arr)) if len(arr) else 0.0,
            "max": float(np.max(arr)) if len(arr) else 0.0,
        },
        "rows": rows,
    }