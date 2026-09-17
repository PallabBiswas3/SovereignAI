from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .root_rank_calibration import cross_validate_root_ranker as _cross_validate_root_ranker


DEFAULT_ROOT_CANDIDATE_LIMIT = 24


def screen_root_candidates(
    records: Iterable[dict[str, Any]],
    *,
    candidate_limit: int = DEFAULT_ROOT_CANDIDATE_LIMIT,
) -> list[dict[str, Any]]:
    """Apply label-blind pre-ranking candidate screening.

    If the feature builder already marked `ranker_eligible`, that fixed data-driven
    shortlist is used exactly. Otherwise we fall back to sorting by shift evidence.
    Audit-only rows remain in the unscreened calibration record for post-mortem
    analysis but never enter cross-validation or prediction.
    """
    limit = max(1, int(candidate_limit))
    screened: list[dict[str, Any]] = []
    for record in records:
        row = deepcopy(record)
        candidates = list(row.get("candidates", []))
        eligible = [item for item in candidates if item.get("ranker_eligible") is True]
        if eligible:
            selected = eligible[:limit]
            method = "precomputed_label_blind_shortlist"
        else:
            candidates.sort(
                key=lambda item: (
                    float(item.get("pre_post_shift_score", 0.0) or 0.0),
                    float(item.get("contribution_score", 0.0) or 0.0),
                    float(item.get("onset_earliness_score", 0.0) or 0.0),
                ),
                reverse=True,
            )
            selected = candidates[:limit]
            method = "label_blind_shift_screen"
        row["candidates"] = selected
        row["candidate_screening"] = {
            "method": method,
            "candidate_limit": limit,
            "pre_screen_count": len(candidates),
            "post_screen_count": len(selected),
        }
        screened.append(row)
    return screened


def cross_validate_root_ranker_screened(
    records: Iterable[dict[str, Any]],
    *,
    candidate_limit: int = DEFAULT_ROOT_CANDIDATE_LIMIT,
    **kwargs,
) -> dict[str, Any]:
    """Run fault-level CV after a fixed, label-blind candidate screen."""
    screened = screen_root_candidates(records, candidate_limit=candidate_limit)
    result = _cross_validate_root_ranker(screened, **kwargs)
    result["candidate_screening"] = {
        "method": "precomputed_or_shift_label_blind_screen",
        "candidate_limit": int(candidate_limit),
    }
    return result
