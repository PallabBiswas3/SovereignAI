from __future__ import annotations

import re
from typing import Any

from controlplane.detectors.base import Detector
from controlplane.schema import (
    DetectorResult,
    Finding,
    FindingStatus,
    Interaction,
    RiskCategory,
    Severity,
)


_DEPENDENCY_RE = re.compile(r"\b(?:as (?:I|we) mentioned earlier|based on (?:that|the previous)|therefore|using the earlier)\b", re.I)


class ConversationRiskDetector(Detector):
    name = "conversation"

    def detect(self, interaction: Interaction, settings: dict[str, Any]) -> DetectorResult:
        prior_scores = [turn.risk_score for turn in interaction.conversation if turn.risk_score is not None]
        metadata_score = interaction.metadata.get("prior_risk_score")
        if isinstance(metadata_score, (int, float)):
            prior_scores.append(float(metadata_score))
        maximum = max(prior_scores, default=0.0)

        findings: list[Finding] = []
        match = _DEPENDENCY_RE.search(interaction.response or "")
        threshold = float(settings.get("prior_risk_threshold", 0.60))
        if match and maximum >= threshold:
            findings.append(
                Finding(
                    category=RiskCategory.POLICY,
                    subtype="compounding_conversation_risk",
                    severity=Severity.HIGH if interaction.consequential else Severity.MEDIUM,
                    confidence=min(1.0, maximum),
                    status=FindingStatus.SUSPECTED,
                    message="The response relies on an earlier turn that carried elevated risk.",
                    detector=self.name,
                    span=match.span(),
                    metadata={"maximum_prior_risk": maximum},
                )
            )
        return DetectorResult(detector=self.name, findings=findings)
