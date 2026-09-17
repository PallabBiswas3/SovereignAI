from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from adaptivefact.data.schema import ResponseLabel, ResponseRecord, VerificationStatus


class RemediationAction(str, Enum):
    """Operational outcome after factual verification."""

    RELEASE = "release"
    CORRECTED_RELEASE = "corrected_release"
    BLOCK = "block"
    SAFE_ABSTAIN = "safe_abstain"
    HUMAN_REVIEW = "human_review"


@dataclass
class RemediationConfig:
    enabled: bool = True
    min_supported_confidence: float = 0.90
    min_supported_claims_for_correction: int = 1
    auto_abstain_nonconsequential: bool = True
    human_review_consequential_uncertainty: bool = True
    max_corrected_response_chars: int = 2_000
    blocked_response: str = (
        "I cannot provide this response because one or more claims conflict "
        "with the available evidence."
    )
    abstention_response: str = (
        "I cannot verify this answer from the approved evidence currently available."
    )
    review_response: str = (
        "This high-impact response requires review because important claims remain unresolved."
    )


@dataclass
class RemediationResult:
    action: RemediationAction
    response: str
    reason: str
    source_claim_ids: list[str]
    removed_claim_ids: list[str]
    human_review_required: bool = False

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["action"] = self.action.value
        return payload


class EvidenceGroundedRemediator:
    """Produce a safe operational response without free-form generation.

    Corrections are extractive: only exact claim text already marked SUPPORTED
    can be retained. This makes the remediation auditable and avoids asking a
    second generative model to invent a replacement answer.
    """

    def __init__(self, config: RemediationConfig | None = None) -> None:
        self.config = config or RemediationConfig()

    def remediate(
        self,
        record: ResponseRecord,
        decision: ResponseLabel,
        *,
        consequential: bool = False,
    ) -> RemediationResult:
        if not self.config.enabled:
            return RemediationResult(
                action=RemediationAction.HUMAN_REVIEW,
                response=self.config.review_response,
                reason="Automatic remediation is disabled.",
                source_claim_ids=[],
                removed_claim_ids=[claim.id for claim in record.atomic_claims],
                human_review_required=True,
            )

        if decision == ResponseLabel.SUPPORTED:
            return RemediationResult(
                action=RemediationAction.RELEASE,
                response=record.generated_response,
                reason="The response is supported and can be released unchanged.",
                source_claim_ids=[claim.id for claim in record.atomic_claims],
                removed_claim_ids=[],
            )

        if decision == ResponseLabel.HALLUCINATED:
            return RemediationResult(
                action=RemediationAction.BLOCK,
                response=self.config.blocked_response,
                reason="At least one claim is contradicted by approved evidence.",
                source_claim_ids=[],
                removed_claim_ids=[claim.id for claim in record.atomic_claims],
            )

        if consequential and self.config.human_review_consequential_uncertainty:
            return RemediationResult(
                action=RemediationAction.HUMAN_REVIEW,
                response=self.config.review_response,
                reason="Consequential unresolved claims require accountable review.",
                source_claim_ids=[],
                removed_claim_ids=[claim.id for claim in record.atomic_claims],
                human_review_required=True,
            )

        supported = [
            claim
            for claim in record.atomic_claims
            if claim.status == VerificationStatus.SUPPORTED
            and (claim.confidence or 0.0) >= self.config.min_supported_confidence
        ]
        if len(supported) >= self.config.min_supported_claims_for_correction:
            retained_text = []
            seen = set()
            for claim in supported:
                normalized = " ".join(claim.text.split()).strip()
                if normalized and normalized.casefold() not in seen:
                    seen.add(normalized.casefold())
                    retained_text.append(normalized)
            corrected = " ".join(retained_text)[: self.config.max_corrected_response_chars].strip()
            if corrected:
                supported_ids = {claim.id for claim in supported}
                return RemediationResult(
                    action=RemediationAction.CORRECTED_RELEASE,
                    response=corrected,
                    reason=(
                        "Unsupported or unresolved claims were removed; only previously "
                        "verified claims were retained."
                    ),
                    source_claim_ids=[claim.id for claim in supported],
                    removed_claim_ids=[
                        claim.id for claim in record.atomic_claims
                        if claim.id not in supported_ids
                    ],
                )

        if self.config.auto_abstain_nonconsequential:
            return RemediationResult(
                action=RemediationAction.SAFE_ABSTAIN,
                response=self.config.abstention_response,
                reason="No sufficiently supported claim was available for a safe correction.",
                source_claim_ids=[],
                removed_claim_ids=[claim.id for claim in record.atomic_claims],
            )

        return RemediationResult(
            action=RemediationAction.HUMAN_REVIEW,
            response=self.config.review_response,
            reason="The response remains unresolved after automatic verification.",
            source_claim_ids=[],
            removed_claim_ids=[claim.id for claim in record.atomic_claims],
            human_review_required=True,
        )
