from __future__ import annotations

from adaptivefact.data.schema import Claim, ClaimType, VerificationStatus


_TYPE_PRIORITY = {
    ClaimType.NUMERIC: 0.90,
    ClaimType.DATE: 0.85,
    ClaimType.CITATION: 0.85,
    ClaimType.ENTITY: 0.70,
    ClaimType.FACTUAL: 0.60,
    ClaimType.OTHER: 0.30,
}


def claim_priority(
    claim: Claim,
    response_risk: float,
    *,
    consequential: bool = False,
) -> float:
    score = _TYPE_PRIORITY.get(claim.type, 0.30)
    method = claim.verifier_used or ""
    if method.endswith("conflict_candidate"):
        score += 0.20
    if claim.numbers or claim.dates:
        score += 0.05
    if consequential:
        score += 0.15
    score += min(1.0, max(0.0, response_risk)) * 0.15
    return min(1.0, score)


def prioritize_unknown_claims(
    claims: list[Claim],
    response_risk: float,
    *,
    consequential: bool = False,
    min_priority: float = 0.0,
    limit: int | None = None,
) -> list[Claim]:
    ranked = []
    for claim in claims:
        if claim.status != VerificationStatus.UNKNOWN:
            continue
        priority = claim_priority(claim, response_risk, consequential=consequential)
        claim.risk_score = priority
        if priority >= min_priority:
            ranked.append((priority, claim))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = [claim for _, claim in ranked]
    return selected if limit is None else selected[: max(0, limit)]
