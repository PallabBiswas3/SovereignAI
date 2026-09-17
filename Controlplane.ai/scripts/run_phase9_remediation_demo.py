#!/usr/bin/env python3
from __future__ import annotations

import json

from adaptivefact.data.schema import Claim, ResponseLabel, ResponseRecord, VerificationStatus
from adaptivefact.pipeline import EvidenceGroundedRemediator


def claim(claim_id: str, text: str, status: VerificationStatus, confidence: float) -> Claim:
    return Claim(id=claim_id, text=text, status=status, confidence=confidence)


def main() -> None:
    remediator = EvidenceGroundedRemediator()
    supported = claim(
        "verified-balance",
        "The verified account balance is $8,000.",
        VerificationStatus.SUPPORTED,
        0.98,
    )
    unknown = claim(
        "unverified-refund",
        "The customer will receive a refund tomorrow.",
        VerificationStatus.UNKNOWN,
        0.60,
    )

    scenarios = [
        (
            "extractive_correction",
            ResponseRecord(
                id="phase9-correct",
                dataset="controlled",
                query="Summarize the verified account status.",
                generated_response=(
                    "The verified account balance is $8,000. "
                    "The customer will receive a refund tomorrow."
                ),
                atomic_claims=[supported.model_copy(deep=True), unknown.model_copy(deep=True)],
            ),
            ResponseLabel.MIXED,
            False,
        ),
        (
            "safe_abstention",
            ResponseRecord(
                id="phase9-abstain",
                dataset="controlled",
                query="When will the refund arrive?",
                generated_response="The refund will arrive tomorrow.",
                atomic_claims=[unknown.model_copy(deep=True)],
            ),
            ResponseLabel.UNKNOWN,
            False,
        ),
        (
            "consequential_review",
            ResponseRecord(
                id="phase9-review",
                dataset="controlled",
                query="Should this loan be denied?",
                generated_response="The loan should be denied.",
                atomic_claims=[
                    claim(
                        "loan-decision",
                        "The loan should be denied.",
                        VerificationStatus.UNKNOWN,
                        0.80,
                    )
                ],
                metadata={"consequential": True},
            ),
            ResponseLabel.UNKNOWN,
            True,
        ),
    ]

    output = []
    for name, record, decision, consequential in scenarios:
        result = remediator.remediate(
            record,
            decision,
            consequential=consequential,
        )
        output.append(
            {
                "scenario": name,
                "original_response": record.generated_response,
                "verification_decision": decision.value,
                "remediation": result.to_dict(),
            }
        )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
