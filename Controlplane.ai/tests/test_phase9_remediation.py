from __future__ import annotations

from adaptivefact.data.schema import (
    Claim,
    ResponseLabel,
    ResponseRecord,
    VerificationStatus,
)
from adaptivefact.pipeline import (
    EvidenceGroundedRemediator,
    RemediationAction,
    RemediationConfig,
)


def _record(*claims: Claim, consequential: bool = False) -> ResponseRecord:
    return ResponseRecord(
        id="remediation-record",
        dataset="synthetic",
        query="What is verified?",
        generated_response="The balance is $8,000. Support is available weekdays.",
        atomic_claims=list(claims),
        metadata={"consequential": consequential},
    )


def _claim(claim_id: str, text: str, status: VerificationStatus, confidence: float) -> Claim:
    return Claim(
        id=claim_id,
        text=text,
        status=status,
        confidence=confidence,
    )


def test_supported_response_is_released_unchanged():
    record = _record(_claim("c1", "The balance is $8,000.", VerificationStatus.SUPPORTED, 0.98))
    result = EvidenceGroundedRemediator().remediate(record, ResponseLabel.SUPPORTED)

    assert result.action == RemediationAction.RELEASE
    assert result.response == record.generated_response


def test_mixed_response_keeps_only_high_confidence_supported_claims():
    supported = _claim(
        "c1",
        "The balance is $8,000.",
        VerificationStatus.SUPPORTED,
        0.98,
    )
    unresolved = _claim(
        "c2",
        "Support is available every day.",
        VerificationStatus.UNKNOWN,
        0.70,
    )
    result = EvidenceGroundedRemediator().remediate(
        _record(supported, unresolved),
        ResponseLabel.MIXED,
    )

    assert result.action == RemediationAction.CORRECTED_RELEASE
    assert result.response == supported.text
    assert result.source_claim_ids == ["c1"]
    assert result.removed_claim_ids == ["c2"]


def test_unresolved_low_risk_response_safely_abstains_without_human():
    unresolved = _claim("c1", "The balance is unknown.", VerificationStatus.UNKNOWN, 0.6)
    result = EvidenceGroundedRemediator().remediate(
        _record(unresolved),
        ResponseLabel.UNKNOWN,
    )

    assert result.action == RemediationAction.SAFE_ABSTAIN
    assert result.human_review_required is False


def test_consequential_uncertainty_requires_human_review():
    unresolved = _claim("c1", "The loan should be denied.", VerificationStatus.UNKNOWN, 0.8)
    result = EvidenceGroundedRemediator().remediate(
        _record(unresolved, consequential=True),
        ResponseLabel.UNKNOWN,
        consequential=True,
    )

    assert result.action == RemediationAction.HUMAN_REVIEW
    assert result.human_review_required is True


def test_contradicted_response_is_automatically_blocked():
    contradicted = _claim(
        "c1",
        "The balance is $12,000.",
        VerificationStatus.CONTRADICTED,
        0.99,
    )
    result = EvidenceGroundedRemediator().remediate(
        _record(contradicted),
        ResponseLabel.HALLUCINATED,
    )

    assert result.action == RemediationAction.BLOCK
    assert result.human_review_required is False


def test_low_confidence_support_is_not_used_for_correction():
    weak = _claim("c1", "The balance is $8,000.", VerificationStatus.SUPPORTED, 0.75)
    remediator = EvidenceGroundedRemediator(
        RemediationConfig(min_supported_confidence=0.9)
    )
    result = remediator.remediate(_record(weak), ResponseLabel.MIXED)

    assert result.action == RemediationAction.SAFE_ABSTAIN
