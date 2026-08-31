from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.rag.embeddings import EmbeddingProvider, configured_embedding_provider


class ClaimStatus(str, Enum):
    supported = "SUPPORTED"
    weakly_supported = "WEAKLY_SUPPORTED"
    unsupported = "UNSUPPORTED"


class ClaimAssessment(BaseModel):
    claim: str
    status: ClaimStatus
    grounded: bool
    confidence: float
    lexical_score: float
    semantic_score: float
    retrieval_score: float | None = None
    judge_score: float | None = None
    source: dict[str, Any] | None = None


class EvidenceSupportReport(BaseModel):
    grounding_confidence: float
    claims: list[ClaimAssessment] = Field(default_factory=list)
    unsupported_material_claims: list[str] = Field(default_factory=list)


class GroundingChecker:
    """Claim-level semantic evidence checker; retrieval alone is never treated as proof."""

    def __init__(self, threshold: float = 0.55, weak_threshold: float = 0.38, embeddings: EmbeddingProvider | None = None) -> None:
        self.threshold = threshold
        self.weak_threshold = min(weak_threshold, threshold)
        self.embeddings = embeddings or configured_embedding_provider()

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stop = {"the", "a", "an", "is", "are", "of", "to", "and", "in", "for", "that", "this", "be", "shall"}
        return {token for token in re.findall(r"[a-z0-9.]+", text.lower()) if token not in stop and len(token) > 1}

    @staticmethod
    def _claims(response: str) -> list[str]:
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+|(?<=;)\s+", response) if len(part.strip()) > 8]

    @staticmethod
    def _dot(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right))))

    @staticmethod
    def _numbers(text: str) -> set[str]:
        return set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", text))

    @staticmethod
    def _material(claim: str) -> bool:
        return bool(re.search(r"\d|must|shall|should|unsafe|safe|limit|approve|remove|replace|failure|critical", claim, re.I))

    def evaluate(self, response: str, sources: list[dict[str, Any]]) -> EvidenceSupportReport:
        claims = self._claims(response)
        source_texts = [str(item.get("text", "")) for item in sources]
        if not claims:
            return EvidenceSupportReport(grounding_confidence=1.0)
        source_vectors = self.embeddings.embed_documents(source_texts) if source_texts else []
        assessments: list[ClaimAssessment] = []
        for claim in claims:
            claim_vector = self.embeddings.embed_query(claim)
            claim_tokens = self._tokens(claim)
            best_combined = 0.0
            best: tuple[float, float, float | None, dict[str, Any] | None] = (0.0, 0.0, None, None)
            for source_index, source in enumerate(sources):
                text = source_texts[source_index]
                lexical = len(claim_tokens & self._tokens(text)) / max(1, len(claim_tokens))
                semantic = self._dot(claim_vector, source_vectors[source_index])
                retrieval_raw = source.get("retrieval_score", source.get("score"))
                retrieval = float(retrieval_raw) if isinstance(retrieval_raw, (int, float)) else None
                numeric_match = not self._numbers(claim) or self._numbers(claim).issubset(self._numbers(text))
                combined = .62 * semantic + .28 * lexical + .10 * (retrieval if retrieval is not None else semantic)
                if not numeric_match:
                    combined *= .35
                if combined > best_combined:
                    best_combined = combined
                    best = (lexical, semantic, retrieval, source)
            lexical, semantic, retrieval, source = best
            if best_combined >= self.threshold:
                status = ClaimStatus.supported
            elif best_combined >= self.weak_threshold:
                status = ClaimStatus.weakly_supported
            else:
                status = ClaimStatus.unsupported
            assessments.append(ClaimAssessment(
                claim=claim, status=status, grounded=status == ClaimStatus.supported,
                confidence=round(best_combined, 3), lexical_score=round(lexical, 3),
                semantic_score=round(semantic, 3), retrieval_score=round(retrieval, 3) if retrieval is not None else None,
                source=source.get("source", source) if source else None,
            ))
        confidence = sum(item.confidence for item in assessments) / len(assessments)
        unsupported = [item.claim for item in assessments if item.status != ClaimStatus.supported and self._material(item.claim)]
        return EvidenceSupportReport(
            grounding_confidence=round(confidence, 3), claims=assessments,
            unsupported_material_claims=unsupported,
        )

    def assess(self, response: str, sources: list[dict[str, Any]]) -> tuple[float, list[ClaimAssessment]]:
        report = self.evaluate(response, sources)
        return report.grounding_confidence, report.claims
