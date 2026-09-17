from __future__ import annotations

from typing import Any, Iterable

from .root_rank_calibration import build_root_feature_rows as _build_root_feature_rows


DEFAULT_ROOT_CANDIDATE_LIMIT = 24


def build_root_feature_rows_screened(
    candidate_names: Iterable[str],
    *,
    contribution_scores: dict[str, float],
    shift_scores: dict[str, float],
    onset_order: Iterable[str],
    process_topology: Any,
    fault_catalog: Any,
    fault_type_scores: dict[str, float] | None = None,
    candidate_limit: int = DEFAULT_ROOT_CANDIDATE_LIMIT,
) -> list[dict[str, Any]]:
    """Build ranker features and audit-only root features without label leakage.

    The actual ranker sees only the strongest data-driven candidates, matching the
    candidate-generation policy that produced the held-out PR #10 result. Global
    catalog roots are nevertheless retained as `ranker_eligible=False` audit rows,
    so every missed true root can be inspected after evaluation. The held-out fault
    label is never used to decide eligibility.
    """
    names = [str(x) for x in candidate_names]
    limit = max(1, int(candidate_limit))
    ordered = sorted(
        names,
        key=lambda name: (
            float(shift_scores.get(name, 0.0) or 0.0),
            float(contribution_scores.get(name, 0.0) or 0.0),
        ),
        reverse=True,
    )
    ranker_names = ordered[:limit]
    eligible = set(ranker_names)

    ranker_rows = _build_root_feature_rows(
        ranker_names,
        contribution_scores=contribution_scores,
        shift_scores=shift_scores,
        onset_order=onset_order,
        process_topology=process_topology,
        fault_catalog=fault_catalog,
        fault_type_scores=fault_type_scores,
    )
    for row in ranker_rows:
        row["ranker_eligible"] = True

    # Audit-only roots get their own full-context features. These rows never enter
    # cross-validation or prediction, but allow complete error post-mortems.
    audit_names = [name for name in names if name not in eligible]
    if audit_names:
        all_rows = _build_root_feature_rows(
            names,
            contribution_scores=contribution_scores,
            shift_scores=shift_scores,
            onset_order=onset_order,
            process_topology=process_topology,
            fault_catalog=fault_catalog,
            fault_type_scores=fault_type_scores,
        )
        for row in all_rows:
            if str(row.get("variable")) in audit_names:
                audit_row = dict(row)
                audit_row["ranker_eligible"] = False
                ranker_rows.append(audit_row)

    return ranker_rows
