from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from adaptivefact.data.schema import Claim, ResponseLabel, VerificationStatus


@dataclass
class AggregationResult:
    label: ResponseLabel
    reason: str
    status_counts: dict[str, int]
    important_unknown_claims: int


def aggregate_claims(
    claims: list[Claim],
    *,
    important_unknown_threshold: float = 0.75,
) -> AggregationResult:
    counts = Counter(claim.status.value for claim in claims)
    contradictions = [claim for claim in claims if claim.status == VerificationStatus.CONTRADICTED]
    unsupported = [claim for claim in claims if claim.status == VerificationStatus.UNSUPPORTED]
    conflicting = [claim for claim in claims if claim.status == VerificationStatus.CONFLICTING]
    unresolved_statuses = {
        VerificationStatus.UNKNOWN,
        VerificationStatus.UNDECIDABLE,
        VerificationStatus.UNVERIFIED,
    }
    important_unknown = [
        claim for claim in claims
        if claim.status in unresolved_statuses and (claim.risk_score or 0.0) >= important_unknown_threshold
    ]

    if contradictions or unsupported:
        return AggregationResult(
            label=ResponseLabel.HALLUCINATED,
            reason=(
                f"{len(contradictions)} contradicted and {len(unsupported)} unsupported claim(s) detected."
            ),
            status_counts=dict(counts),
            important_unknown_claims=len(important_unknown),
        )
    if conflicting:
        return AggregationResult(
            label=ResponseLabel.UNKNOWN,
            reason=f"{len(conflicting)} claim(s) have materially conflicting evidence.",
            status_counts=dict(counts),
            important_unknown_claims=len(important_unknown),
        )
    if important_unknown:
        return AggregationResult(
            label=ResponseLabel.UNKNOWN,
            reason=f"{len(important_unknown)} important claim(s) remain undecidable.",
            status_counts=dict(counts),
            important_unknown_claims=len(important_unknown),
        )
    if claims and all(claim.status == VerificationStatus.SUPPORTED for claim in claims):
        return AggregationResult(
            label=ResponseLabel.SUPPORTED,
            reason="All extracted claims are supported.",
            status_counts=dict(counts),
            important_unknown_claims=0,
        )
    if claims:
        return AggregationResult(
            label=ResponseLabel.MIXED,
            reason="No contradiction was confirmed, but some claims remain unresolved.",
            status_counts=dict(counts),
            important_unknown_claims=0,
        )
    return AggregationResult(
        label=ResponseLabel.UNKNOWN,
        reason="No verifiable claims were extracted.",
        status_counts={},
        important_unknown_claims=0,
    )
