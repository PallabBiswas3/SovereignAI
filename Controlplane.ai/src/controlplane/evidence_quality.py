from __future__ import annotations

from dataclasses import dataclass

from adaptivefact.data.schema import Claim
from controlplane.schema import GroundingEvidence, Interaction


@dataclass
class EvidenceAssessment:
    quality: str
    complete_enough_for_unsupported: bool
    selected_texts: list[str]
    selected_ids: list[str]
    max_retrieval_score: float | None
    reason: str


@dataclass
class EvidenceQualityConfig:
    top_k: int = 4
    strong_retrieval_score: float = 0.35
    minimum_typed_chunks_for_completeness: int = 2
    minimum_overlap_tokens: int = 1


class EvidenceQualityGate:
    """Rank structured GraphRAG evidence and determine evidentiary sufficiency.

    The gate is deliberately conservative. Free-text legacy context can support
    or contradict via NLI, but absence of support in legacy context is not enough
    to call a claim UNSUPPORTED because document completeness/provenance is
    unknown. Structured evidence can become 'complete enough' only when there are
    multiple authorized/provenanced chunks or another explicit completeness
    signal supplied by the caller.
    """

    def __init__(self, config: EvidenceQualityConfig | None = None) -> None:
        self.config = config or EvidenceQualityConfig()

    def assess(self, interaction: Interaction, claim: Claim) -> EvidenceAssessment:
        if not interaction.grounding_evidence:
            if interaction.context and interaction.context.strip():
                return EvidenceAssessment(
                    quality="legacy_context",
                    complete_enough_for_unsupported=False,
                    selected_texts=[interaction.context.strip()],
                    selected_ids=[],
                    max_retrieval_score=None,
                    reason="Legacy context is available but has no structured completeness/provenance signal.",
                )
            return EvidenceAssessment(
                quality="no_evidence",
                complete_enough_for_unsupported=False,
                selected_texts=[],
                selected_ids=[],
                max_retrieval_score=None,
                reason="No grounding evidence was supplied.",
            )

        ranked = sorted(
            interaction.grounding_evidence,
            key=lambda item: self._rank(item, claim),
            reverse=True,
        )[: max(1, self.config.top_k)]
        selected = [item for item in ranked if self._rank(item, claim) > 0]
        if not selected:
            return EvidenceAssessment(
                quality="irrelevant_evidence",
                complete_enough_for_unsupported=False,
                selected_texts=[],
                selected_ids=[],
                max_retrieval_score=max(
                    (item.retrieval_score for item in interaction.grounding_evidence if item.retrieval_score is not None),
                    default=None,
                ),
                reason="Structured evidence exists but none is sufficiently related to the claim.",
            )

        max_score = max((item.retrieval_score or 0.0) for item in selected)
        explicit_complete = any(bool(item.metadata.get("evidence_set_complete")) for item in selected)
        strong_provenance = sum(
            bool(item.document_id or item.source_name) and bool(item.chunk_id or item.page is not None)
            for item in selected
        )
        complete = explicit_complete or (
            len(selected) >= self.config.minimum_typed_chunks_for_completeness
            and strong_provenance >= self.config.minimum_typed_chunks_for_completeness
            and max_score >= self.config.strong_retrieval_score
        )
        return EvidenceAssessment(
            quality="strong_structured" if complete else "partial_structured",
            complete_enough_for_unsupported=complete,
            selected_texts=[item.text for item in selected],
            selected_ids=[item.evidence_id for item in selected],
            max_retrieval_score=max_score,
            reason=(
                "Structured evidence is sufficiently strong/provenanced for absence-of-support reasoning."
                if complete
                else "Structured evidence is useful for support/contradiction checks but not complete enough to infer unsupportedness."
            ),
        )

    def _rank(self, evidence: GroundingEvidence, claim: Claim) -> float:
        text_tokens = set(self._tokens(evidence.text))
        claim_tokens = set(self._tokens(claim.verification_text))
        overlap = len(text_tokens & claim_tokens)
        lexical = overlap / max(1, len(claim_tokens))
        supplied = min(1.0, max(0.0, float(evidence.retrieval_score or 0.0)))
        provenance = 0.05 if evidence.document_id or evidence.source_name else 0.0
        authorized = 0.05 if evidence.authorization_scope else 0.0
        if overlap < self.config.minimum_overlap_tokens and supplied <= 0:
            return 0.0
        return 0.55 * supplied + 0.35 * lexical + provenance + authorized

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.casefold().strip(".,:;()[]{}\"'") for token in text.split() if len(token.strip()) >= 3]
