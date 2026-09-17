from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from adaptivefact.agents.schema import AgentVerificationConfig
from adaptivefact.agents.tools.context import ContextSearchTool
from adaptivefact.agents.verifier import BoundedVerificationAgent
from adaptivefact.data.schema import GenerationMetadata, ResponseLabel, ResponseRecord, VerificationStatus
from adaptivefact.extraction.claim_extractor import ClaimExtractionConfig
from adaptivefact.pipeline.aggregation import aggregate_claims
from adaptivefact.verification.deterministic import DeterministicVerifierConfig
from adaptivefact.verification.evidence import EvidenceRetrieverConfig
from adaptivefact.verification.nli import NLIScorer, TransformersNLIScorer
from adaptivefact.verification.phase5 import Phase5Pipeline
from adaptivefact.verification.phase6 import Phase6NLIConfig, Phase6Pipeline
from controlplane.schema import (
    DetectorResult,
    Finding,
    FindingStatus,
    Interaction,
    RiskCategory,
    Severity,
    VerificationDepth,
)
from controlplane.detectors.privacy import mask_privacy_values


class VerificationService(Protocol):
    def verify(self, interaction: Interaction, depth: VerificationDepth) -> DetectorResult: ...


class LazyTransformersNLIScorer(NLIScorer):
    """Load the configured NLI model on the first deep-verification request."""

    def __init__(self, model_name: str, *, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._delegate: TransformersNLIScorer | None = None

    def score(self, premises: list[str], hypotheses: list[str]):
        if self._delegate is None:
            self._delegate = TransformersNLIScorer(self.model_name, device=self.device)
        return self._delegate.score(premises, hypotheses)


@dataclass
class AdaptiveVerificationConfig:
    standard_nli_claim_limit: int = 8
    deep_nli_claim_limit: int = 20
    deep_agent_claim_limit: int = 5
    important_unknown_threshold: float = 0.75


class AdaptiveFactVerificationService:
    """Depth-aware bridge from ControlPlane interactions to AdaptiveFact.

    QUICK executes Phase 5 only. STANDARD adds retrieval/NLI. DEEP adds the
    bounded evidence agent. Results are normalized as ControlPlane findings;
    the final enforcement decision remains owned by the policy engine.
    """

    name = "adaptivefact"

    def __init__(
        self,
        *,
        nli: NLIScorer | None = None,
        agent: BoundedVerificationAgent | None = None,
        config: AdaptiveVerificationConfig | None = None,
        extraction_config: ClaimExtractionConfig | None = None,
        deterministic_config: DeterministicVerifierConfig | None = None,
        retrieval_config: EvidenceRetrieverConfig | None = None,
        nli_config: Phase6NLIConfig | None = None,
    ) -> None:
        self.agent = agent
        self.config = config or AdaptiveVerificationConfig()
        self.phase5 = Phase5Pipeline(extraction_config, deterministic_config)
        self.phase6 = (
            Phase6Pipeline(nli, retrieval_config=retrieval_config, nli_config=nli_config)
            if nli is not None
            else None
        )

    def verify(self, interaction: Interaction, depth: VerificationDepth) -> DetectorResult:
        started = perf_counter()
        factuality_response = mask_privacy_values(interaction.response)
        record = ResponseRecord(
            id=interaction.id,
            dataset="controlplane_live",
            query=interaction.prompt,
            context=interaction.context,
            generated_response=factuality_response,
            generation_metadata=GenerationMetadata(model=interaction.metadata.get("model")),
            metadata={"consequential": interaction.consequential},
        )
        _, phase5_runtime = self.phase5.process(record)
        phase6_runtime: dict = {}
        agent_runtime = {"claims": 0, "search_calls": 0, "nli_pairs": 0, "cost": 0.0, "latency_ms": 0.0}

        if depth in {VerificationDepth.STANDARD, VerificationDepth.DEEP}:
            if self.phase6 is None:
                raise RuntimeError(f"{depth.value} verification requires a configured NLI scorer")
            unknown = [claim for claim in record.atomic_claims if claim.status == VerificationStatus.UNKNOWN]
            limit = self.config.standard_nli_claim_limit if depth == VerificationDepth.STANDARD else self.config.deep_nli_claim_limit
            selected = unknown[: max(0, limit)]
            _, phase6_runtime = self.phase6.process_records(
                [record], claim_ids={claim.id for claim in selected}
            )

        if depth == VerificationDepth.DEEP:
            if self.agent is None:
                raise RuntimeError("deep verification requires a configured bounded agent")
            candidates = [claim for claim in record.atomic_claims if claim.status == VerificationStatus.UNKNOWN][
                : max(0, self.config.deep_agent_claim_limit)
            ]
            results = []
            for claim in candidates:
                result = self.agent.verify(record, claim)
                self.agent.apply(claim, result)
                results.append(result)
            agent_runtime = {
                "claims": len(results),
                "search_calls": sum(item.search_calls for item in results),
                "nli_pairs": sum(item.nli_pairs for item in results),
                "cost": sum(item.total_cost for item in results),
                "latency_ms": sum(item.latency_ms for item in results),
            }

        aggregation = aggregate_claims(
            record.atomic_claims,
            important_unknown_threshold=self.config.important_unknown_threshold,
        )
        findings = self._normalize_findings(record, aggregation.label, interaction=interaction, depth=depth)
        return DetectorResult(
            detector=self.name,
            findings=findings,
            latency_ms=(perf_counter() - started) * 1000.0,
            metadata={
                "verification_depth": depth.value,
                "response_status": aggregation.label.value,
                "status_counts": aggregation.status_counts,
                "claims_checked": len(record.atomic_claims),
                "phase5": phase5_runtime,
                "phase6": phase6_runtime,
                "agent": agent_runtime,
                "claim_results": [
                    {
                        "claim_id": claim.id,
                        "claim": claim.text,
                        "status": claim.status.value,
                        "confidence": claim.confidence,
                        "verifier": claim.verifier_used,
                        "nli_scores": claim.verification_scores,
                        "evidence": claim.evidence,
                    }
                    for claim in record.atomic_claims
                ],
            },
        )

    def _normalize_findings(
        self,
        record: ResponseRecord,
        response_status: ResponseLabel,
        *,
        interaction: Interaction,
        depth: VerificationDepth,
    ) -> list[Finding]:
        findings: list[Finding] = []
        for claim in record.atomic_claims:
            if claim.status == VerificationStatus.SUPPORTED:
                continue
            metadata = {
                "claim_id": claim.id,
                "claim": claim.text,
                "claim_type": claim.type.value,
                "verifier": claim.verifier_used,
                "response_status": response_status.value,
                "verification_depth": depth.value,
                "nli_scores": claim.verification_scores,
            }
            if claim.status == VerificationStatus.CONTRADICTED:
                findings.append(
                    Finding(
                        category=RiskCategory.HALLUCINATION,
                        subtype="claim_contradicted",
                        severity=Severity.HIGH,
                        confidence=float(claim.confidence or 0.5),
                        status=FindingStatus.CONTRADICTED,
                        message="A factual claim is contradicted by retrieved evidence.",
                        detector=self.name,
                        span=claim.span,
                        evidence=claim.evidence,
                        metadata=metadata,
                    )
                )
            elif claim.status in {VerificationStatus.UNKNOWN, VerificationStatus.UNVERIFIED}:
                findings.append(
                    Finding(
                        category=RiskCategory.HALLUCINATION,
                        subtype="claim_unknown",
                        severity=Severity.HIGH if interaction.consequential else Severity.MEDIUM,
                        confidence=float(claim.confidence or 0.5),
                        status=FindingStatus.UNKNOWN,
                        message="A factual claim remains unresolved after the configured verification route.",
                        detector=self.name,
                        span=claim.span,
                        evidence=claim.evidence,
                        metadata=metadata,
                    )
                )
        return findings


def build_adaptive_verification_from_env() -> AdaptiveFactVerificationService | None:
    """Build adaptive verification unless it is explicitly disabled."""

    enabled = os.getenv("CONTROLPLANE_ADAPTIVE_VERIFICATION", "1").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    model_name = os.getenv("CONTROLPLANE_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
    device = os.getenv("CONTROLPLANE_NLI_DEVICE") or None
    nli = LazyTransformersNLIScorer(model_name, device=device)
    retrieval = EvidenceRetrieverConfig(min_score=0.05)
    agent = BoundedVerificationAgent(
        nli,
        [ContextSearchTool(retrieval_config=retrieval)],
        AgentVerificationConfig(),
    )
    return AdaptiveFactVerificationService(nli=nli, agent=agent, retrieval_config=retrieval)
