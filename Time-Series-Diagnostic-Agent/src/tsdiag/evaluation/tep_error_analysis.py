from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


FAILURE_CATEGORIES = (
    "DOWNSTREAM_OVER_WEIGHTING",
    "EARLY_NOISE_ONSET",
    "CATALOG_AMBIGUITY",
    "WEAK_SIGNAL",
)

COMPONENT_NAMES = (
    "contribution_score",
    "pre_post_shift_score",
    "onset_earliness_score",
    "topology_upstreamness_score",
    "fault_type_agreement_score",
    "catalog_root_prior_score",
)


def _candidate_map(record: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        str(row["variable"]): {
            key: float(row.get(key, 0.0) or 0.0)
            for key in COMPONENT_NAMES
        }
        for row in record.get("candidates", [])
        if "variable" in row
    }


def _missing_components() -> dict[str, None]:
    return {name: None for name in COMPONENT_NAMES}


def _score_with_weights(features: dict[str, float], weights: dict[str, float]) -> float:
    return float(sum(float(weights.get(name, 0.0)) * float(value) for name, value in features.items()))


def _best_true_root(expected_roots: Iterable[str], feature_map, weights):
    choices = []
    for root in expected_roots:
        if root in feature_map:
            choices.append((root, _score_with_weights(feature_map[root], weights)))
    if not choices:
        return None, None
    choices.sort(key=lambda item: item[1], reverse=True)
    return choices[0]


def _failure_category(
    true_features: dict[str, float],
    wrong_features: dict[str, float],
    root_support_gap: float,
) -> str:
    if true_features.get("pre_post_shift_score", 0.0) < 0.20:
        return "WEAK_SIGNAL"
    if (
        wrong_features.get("onset_earliness_score", 0.0)
        - true_features.get("onset_earliness_score", 0.0)
        >= 0.30
    ):
        return "EARLY_NOISE_ONSET"
    if (
        abs(
            wrong_features.get("catalog_root_prior_score", 0.0)
            - true_features.get("catalog_root_prior_score", 0.0)
        ) <= 0.10
        and abs(
            wrong_features.get("fault_type_agreement_score", 0.0)
            - true_features.get("fault_type_agreement_score", 0.0)
        ) <= 0.15
    ):
        return "CATALOG_AMBIGUITY"
    if (
        wrong_features.get("contribution_score", 0.0)
        + wrong_features.get("pre_post_shift_score", 0.0)
        > true_features.get("contribution_score", 0.0)
        + true_features.get("pre_post_shift_score", 0.0)
        and true_features.get("topology_upstreamness_score", 0.0)
        >= wrong_features.get("topology_upstreamness_score", 0.0)
    ):
        return "DOWNSTREAM_OVER_WEIGHTING"
    if root_support_gap < 0.08:
        return "CATALOG_AMBIGUITY"
    return "DOWNSTREAM_OVER_WEIGHTING"


def analyze_root_cause_errors(
    calibration_records: Iterable[dict[str, Any]],
    held_out_predictions: Iterable[dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, Any]:
    """Post-mortem every held-out Top-1 mistake.

    The report intentionally keeps candidate-generation failures. If none of the
    expected roots survived the label-blind candidate screen, the mistake is
    recorded as WEAK_SIGNAL with `true_root_in_candidates=false` rather than being
    silently dropped. This keeps error_count equal to the number of held-out Top-1
    misses and makes candidate recall itself auditable.
    """
    records = {int(row["fault_id"]): row for row in calibration_records}
    predictions = {int(row["fault_id"]): row for row in held_out_predictions}
    errors = []

    for fault_id, prediction in sorted(predictions.items()):
        record = records.get(fault_id)
        if record is None:
            continue
        expected = [str(x) for x in record.get("expected_roots", [])]
        if not expected:
            continue

        raw_top = prediction.get("raw_top_root")
        if raw_top in expected:
            continue

        feature_map = _candidate_map(record)
        ranked = prediction.get("ranking", [])
        wrong = str(raw_top) if raw_top is not None else None
        wrong_features = feature_map.get(wrong, _missing_components()) if wrong else _missing_components()
        wrong_score = None
        if wrong is not None:
            ranked_wrong = next((row for row in ranked if row.get("variable") == wrong), None)
            if ranked_wrong is not None:
                wrong_score = float(ranked_wrong.get("score", 0.0))
            elif wrong in feature_map:
                wrong_score = _score_with_weights(feature_map[wrong], weights)

        true_candidates = [row for row in ranked if row.get("variable") in expected]
        true_candidates.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        true_row = true_candidates[0] if true_candidates else None

        if true_row is not None:
            true_root = str(true_row["variable"])
            true_score = float(true_row.get("score", 0.0))
            true_rank = next(
                (i + 1 for i, row in enumerate(ranked) if row.get("variable") == true_root),
                None,
            )
            true_features = feature_map.get(true_root, _missing_components())
            true_in_candidates = True
        else:
            true_root, true_score = _best_true_root(expected, feature_map, weights)
            if true_root is not None:
                true_features = feature_map[true_root]
                true_in_candidates = True
            else:
                # No expected root survived the ranker's candidate screen. Preserve
                # the fault as an explicit post-mortem rather than omitting it.
                true_root = expected[0]
                true_score = None
                true_features = _missing_components()
                true_in_candidates = False
            true_rank = None

        if true_score is None or wrong_score is None:
            gap = None
            category = "WEAK_SIGNAL"
        else:
            gap = float(wrong_score - float(true_score))
            category = _failure_category(true_features, wrong_features, gap)

        errors.append(
            {
                "fault_id": fault_id,
                "expected_roots": expected,
                "true_root": true_root,
                "true_root_rank": true_rank,
                "true_root_in_candidates": true_in_candidates,
                "top_wrong_variable": wrong,
                "root_support_gap": gap,
                "primary_failure_cause": category,
                "true_root_score": true_score,
                "top_wrong_score": wrong_score,
                "true_root_components": true_features,
                "top_wrong_components": wrong_features,
                "abstained": bool(prediction.get("abstained")),
                "score_margin": float(prediction.get("score_margin", 0.0) or 0.0),
            }
        )

    counts = Counter(row["primary_failure_cause"] for row in errors)
    candidate_omissions = sum(not row["true_root_in_candidates"] for row in errors)
    return {
        "error_count": len(errors),
        "candidate_omission_count": int(candidate_omissions),
        "category_counts": {name: int(counts.get(name, 0)) for name in FAILURE_CATEGORIES},
        "errors": errors,
    }


def write_error_analysis_json(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def write_error_analysis_markdown(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TEP Root-Cause Error Analysis",
        "",
        f"Held-out Top-1 mistakes: **{report.get('error_count', 0)}**",
        f"True-root candidate omissions: **{report.get('candidate_omission_count', 0)}**",
        "",
        "## Error taxonomy",
        "",
    ]
    for category in FAILURE_CATEGORIES:
        lines.append(f"- **{category}**: {report.get('category_counts', {}).get(category, 0)}")
    lines.extend(
        [
            "",
            "## Misdiagnosed faults",
            "",
            "| Fault | True root | In candidates | True rank | Top wrong variable | Support gap | Primary failure cause |",
            "| ---: | --- | --- | ---: | --- | ---: | --- |",
        ]
    )
    for row in report.get("errors", []):
        rank = "-" if row.get("true_root_rank") is None else str(row["true_root_rank"])
        gap = "-" if row.get("root_support_gap") is None else f"{row['root_support_gap']:.4f}"
        lines.append(
            f"| {row['fault_id']} | {row['true_root']} | {row['true_root_in_candidates']} | {rank} | "
            f"{row.get('top_wrong_variable') or '-'} | {gap} | {row['primary_failure_cause']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
