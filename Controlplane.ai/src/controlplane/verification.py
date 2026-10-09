from __future__ import annotations

import os
import math
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from adaptivefact.agents.schema import AgentVerificationConfig
from adaptivefact.agents.tools.context import ContextSearchTool
from adaptivefact.agents.verifier import BoundedVerificationAgent
from adaptivefact.data.schema import ResponseLabel, VerificationStatus
from adaptivefact.extraction.claim_extractor import ClaimExtractionConfig
from adaptivefact.pipeline.aggregation import aggregate_claims
from adaptivefact.verification.deterministic import DeterministicVerifierConfig
from adaptivefact.verification.evidence import EvidenceRetrieverConfig
from adaptivefact.verification.nli import NLIScorer, TransformersNLIScorer
from adaptivefact.verification.phase5 import Phase5Pipeline
from adaptivefact.verification.phase6 import Phase6NLIConfig, Phase6Pipeline
from controlplane.detectors.privacy import mask_privacy_values
from controlplane.evidence_authority import apply_evidence_authority_checks
from controlplane.evidence_quality import EvidenceAssessment, EvidenceQualityConfig, EvidenceQualityGate
from controlplane.factuality import PreparedFactuality, build_response_record
from controlplane.schema import DetectorResult, Finding, FindingStatus, Interaction, RiskCategory, Severity, VerificationDepth
from controlplane.support_scorers import AlignScoreSupportScorer, MiniCheckSupportScorer, NLIEntailmentSupportScorer, SupportScorer


_SOURCE_CITATION = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9:_-]{2,127})\]")


class VerificationService(Protocol):
    def verify(self, interaction: Interaction, depth: VerificationDepth, prepared: PreparedFactuality | None = None) -> DetectorResult: ...


class LazyTransformersNLIScorer(NLIScorer):
    def __init__(self, model_name: str, *, device: str | None = None, local_files_only: bool = True) -> None:
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self._delegate: TransformersNLIScorer | None = None

    def score(self, premises: list[str], hypotheses: list[str]):
        if self._delegate is None:
            self._delegate = TransformersNLIScorer(
                self.model_name, device=self.device, local_files_only=self.local_files_only,
            )
        return self._delegate.score(premises, hypotheses)


@dataclass
class AdaptiveVerificationConfig:
    standard_nli_claim_limit: int = 8
    deep_nli_claim_limit: int = 20
    deep_agent_claim_limit: int = 5
    important_unknown_threshold: float = 0.75
    support_supported_threshold: float = 0.75
    support_unsupported_threshold: float = 0.25


class AdaptiveFactVerificationService:
    """Evidence-grounded factuality verification with explicit abstention.

    DeBERTa/NLI remains the contradiction-capable checker. Optional MiniCheck or
    Align-style scorers estimate support only. Low support becomes UNSUPPORTED
    only when the structured evidence set is sufficiently complete; otherwise
    the claim becomes UNDECIDABLE. Support-model inference is batched once per
    response to avoid a per-claim model-call latency multiplier.
    """

    name = "adaptivefact"

    def __init__(
        self,
        *,
        nli: NLIScorer | None = None,
        agent: BoundedVerificationAgent | None = None,
        support_scorer: SupportScorer | None = None,
        config: AdaptiveVerificationConfig | None = None,
        extraction_config: ClaimExtractionConfig | None = None,
        deterministic_config: DeterministicVerifierConfig | None = None,
        retrieval_config: EvidenceRetrieverConfig | None = None,
        nli_config: Phase6NLIConfig | None = None,
        evidence_quality_config: EvidenceQualityConfig | None = None,
    ) -> None:
        self.agent = agent
        self.support_scorer = support_scorer
        self.config = config or AdaptiveVerificationConfig()
        self._phase5_customized = extraction_config is not None or deterministic_config is not None
        self.phase5 = Phase5Pipeline(extraction_config, deterministic_config)
        self.phase6 = Phase6Pipeline(nli, retrieval_config=retrieval_config, nli_config=nli_config) if nli is not None else None
        self.evidence_gate = EvidenceQualityGate(evidence_quality_config)

    def verify(
        self,
        interaction: Interaction,
        depth: VerificationDepth,
        prepared: PreparedFactuality | None = None,
        *,
        evidence_authority_enabled: bool = True,
    ) -> DetectorResult:
        started = perf_counter()
        if prepared is None or self._phase5_customized:
            record = build_response_record(interaction, factuality_response=mask_privacy_values(interaction.response))
            _, phase5_runtime = self.phase5.process(record)
            phase5_reused = False
        else:
            record = prepared.record.model_copy(deep=True)
            phase5_runtime = dict(prepared.phase5_runtime)
            phase5_reused = True

        phase6_runtime: dict = {}
        agent_runtime = {"claims": 0, "search_calls": 0, "nli_pairs": 0, "cost": 0.0, "latency_ms": 0.0}

        if depth in {VerificationDepth.STANDARD, VerificationDepth.DEEP}:
            if self.phase6 is None:
                raise RuntimeError(f"{depth.value} verification requires a configured NLI scorer")

            # Structured multi-source evidence must be allowed to challenge an
            # early deterministic support decision. Legacy free-text context keeps
            # v2 behavior and only sends unresolved claims to NLI.
            recheck_supported = bool(interaction.grounding_evidence)
            eligible = [
                claim
                for claim in record.atomic_claims
                if claim.status == VerificationStatus.UNKNOWN
                or (recheck_supported and claim.status == VerificationStatus.SUPPORTED)
            ]
            limit = self.config.standard_nli_claim_limit if depth == VerificationDepth.STANDARD else self.config.deep_nli_claim_limit
            selected = eligible[: max(0, limit)]
            _, phase6_runtime = self.phase6.process_records(
                [record],
                claim_ids={claim.id for claim in selected},
                recheck_supported=recheck_supported,
            )

        if depth == VerificationDepth.DEEP:
            if self.agent is None:
                raise RuntimeError("deep verification requires a configured bounded agent")
            candidates = [claim for claim in record.atomic_claims if claim.status == VerificationStatus.UNKNOWN][: max(0, self.config.deep_agent_claim_limit)]
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

        # Reuse the entailment already computed by phase 6 for auditability.
        # This is a signal, not a calibrated support probability or release
        # decision: source authority and evidence quality are checked below.
        for claim in record.atomic_claims:
            entailment = claim.verification_scores.get("entailment")
            if entailment is not None:
                claim.verification_scores["support_score"] = entailment
                claim.verification_metadata["support_score_basis"] = "phase6_nli_entailment_signal"

        support_started = perf_counter()
        assessments = {claim.id: self.evidence_gate.assess(interaction, claim) for claim in record.atomic_claims}
        support_runtime = self._finalize_claim_states(
            record.atomic_claims, assessments, depth=depth, interaction=interaction,
        )
        support_runtime["latency_ms"] = (perf_counter() - support_started) * 1000.0
        authority_runtime = (
            apply_evidence_authority_checks(interaction, record.atomic_claims)
            if evidence_authority_enabled else {"disabled_for_ablation": True}
        )

        aggregation = aggregate_claims(record.atomic_claims, important_unknown_threshold=self.config.important_unknown_threshold)
        findings = self._normalize_findings(record, aggregation.label, interaction=interaction, depth=depth, assessments=assessments)
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
                "phase5_reused": phase5_reused,
                "phase6": phase6_runtime,
                "agent": agent_runtime,
                "support": support_runtime,
                "evidence_authority": authority_runtime,
                "evidence_authority_enabled": evidence_authority_enabled,
                "claim_results": [
                    {
                        "claim_id": claim.id,
                        "claim": claim.text,
                        "verification_text": claim.verification_text,
                        "structure": {
                            "subject": claim.subject,
                            "predicate": claim.predicate,
                            "object": claim.object,
                            "qualifiers": claim.qualifiers,
                        },
                        "status": claim.status.value,
                        "confidence": claim.confidence,
                        "verifier": claim.verifier_used,
                        "verification_scores": claim.verification_scores,
                        # Backward-compatible alias used by factuality-v2
                        # integration tests and downstream consumers.
                        "nli_scores": claim.verification_scores,
                        "evidence": claim.evidence,
                        "evidence_ids": claim.evidence_ids,
                        "evidence_quality": assessments[claim.id].quality,
                        "evidence_complete_enough": assessments[claim.id].complete_enough_for_unsupported,
                        "authority_reason": claim.verification_metadata.get("authority_reason"),
                        "support_score_basis": claim.verification_metadata.get("support_score_basis"),
                    }
                    for claim in record.atomic_claims
                ],
            },
        )

    def _finalize_claim_states(
        self, claims, assessments: dict[str, EvidenceAssessment],
        *, depth: VerificationDepth, interaction: Interaction,
    ) -> dict:
        backend = getattr(self.support_scorer, "name", None)
        batch_claims = []
        batch_documents: list[str] = []
        batch_texts: list[str] = []
        batch_ids: list[list[str]] = []
        batch_can_promote: list[bool] = []
        evidence_by_id = {item.evidence_id: item for item in interaction.grounding_evidence}

        for claim in claims:
            assessment = assessments[claim.id]
            if claim.status in {VerificationStatus.SUPPORTED, VerificationStatus.CONTRADICTED}:
                continue
            method = claim.verifier_used or ""
            if method == "nli_conflicting_evidence" or method == "agent:conflicting_evidence":
                claim.status = VerificationStatus.CONFLICTING
                claim.verifier_used = f"{method}:v3"
                claim.verification_metadata["evidence_quality"] = assessment.quality
                continue
            if assessment.quality in {"no_evidence", "irrelevant_evidence"}:
                self._mark_undecidable(claim, assessment, "v3_evidence_insufficient")
                continue
            selected_texts = assessment.selected_texts
            selected_ids = assessment.selected_ids
            cited_ids = set(_SOURCE_CITATION.findall(claim.verification_text))
            if evidence_by_id:
                selected_sources = [evidence_by_id[item_id] for item_id in selected_ids if item_id in evidence_by_id]
                if cited_ids:
                    selected_sources = [item for item in selected_sources if item.evidence_id in cited_ids]
                    if {item.evidence_id for item in selected_sources} != cited_ids:
                        selected_sources = []
                selected_sources = [
                    item for item in selected_sources
                    if str(item.metadata.get("document_status", "")).casefold()
                    not in {"superseded", "unverified", "draft"}
                ]
                selected_texts = [item.text for item in selected_sources]
                selected_ids = [item.evidence_id for item in selected_sources]
            if self.support_scorer is not None and depth in {VerificationDepth.STANDARD, VerificationDepth.DEEP} and selected_texts:
                batch_claims.append(claim)
                batch_documents.append("\n\n".join(selected_texts))
                batch_texts.append(_SOURCE_CITATION.sub("", claim.verification_text).strip())
                batch_ids.append(selected_ids)
                batch_can_promote.append(backend != "nli_entailment" or bool(cited_ids))
                continue
            self._mark_undecidable(claim, assessment, method or "phase5_unresolved")

        scores: list[float] = []
        if batch_claims:
            scores = [float(value) for value in self.support_scorer.score(batch_documents, batch_texts)]  # type: ignore[union-attr]
            if len(scores) != len(batch_claims):
                raise RuntimeError("support scorer returned a different number of scores than requested claims")
            for claim, score, source_ids, can_promote in zip(batch_claims, scores, batch_ids, batch_can_promote):
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise RuntimeError("support scorer returned a non-finite or out-of-range score")
                assessment = assessments[claim.id]
                claim.verification_scores["support_score"] = score
                claim.verification_metadata["support_score_basis"] = f"support:{backend}"
                claim.evidence = [evidence_by_id[item_id].text for item_id in source_ids] if evidence_by_id else assessment.selected_texts
                claim.evidence_ids = source_ids
                claim.verification_metadata.update({"support_backend": backend, "evidence_quality": assessment.quality})
                if can_promote and score >= self.config.support_supported_threshold:
                    claim.status = VerificationStatus.SUPPORTED
                    claim.confidence = score
                    claim.verifier_used = f"support:{backend}:supported"
                elif (backend != "nli_entailment" and score <= self.config.support_unsupported_threshold
                      and assessment.complete_enough_for_unsupported):
                    claim.status = VerificationStatus.UNSUPPORTED
                    claim.confidence = 1.0 - score
                    claim.verifier_used = f"support:{backend}:unsupported"
                else:
                    self._mark_undecidable(claim, assessment, f"support:{backend}:inconclusive")

        return {
            "backend": backend,
            "claims": len(batch_claims),
            "calls": 1 if batch_claims else 0,
            "scores": scores,
        }

    @staticmethod
    def _mark_undecidable(claim, assessment: EvidenceAssessment, method: str) -> None:
        claim.status = VerificationStatus.UNDECIDABLE
        claim.verifier_used = f"{method}:undecidable_v3"
        claim.verification_metadata.update({"evidence_quality": assessment.quality, "reason": assessment.reason})

    @staticmethod
    def _legacy_unknown_reason(claim, assessment: EvidenceAssessment) -> str:
        method = claim.verifier_used or ""
        if assessment.quality == "no_evidence":
            return "no_evidence"
        if "conflict_candidate" in method:
            return "deterministic_conflict_candidate"
        if "support:" in method:
            return "insufficient_support"
        return "verifier_uncertain"

    def _normalize_findings(self, record, response_status: ResponseLabel, *, interaction: Interaction, depth: VerificationDepth, assessments: dict[str, EvidenceAssessment]) -> list[Finding]:
        findings: list[Finding] = []
        for claim in record.atomic_claims:
            if claim.status == VerificationStatus.SUPPORTED:
                continue
            assessment = assessments[claim.id]
            metadata = {
                "claim_id": claim.id,
                "claim": claim.text,
                "verification_text": claim.verification_text,
                "claim_type": claim.type.value,
                "verifier": claim.verifier_used,
                "response_status": response_status.value,
                "verification_depth": depth.value,
                "verification_scores": claim.verification_scores,
                "nli_scores": claim.verification_scores,
                "evidence_quality": assessment.quality,
                "evidence_complete_enough": assessment.complete_enough_for_unsupported,
                "evidence_reason": assessment.reason,
                "evidence_ids": claim.evidence_ids or assessment.selected_ids,
                "authority_reason": claim.verification_metadata.get("authority_reason"),
            }
            if claim.status == VerificationStatus.CONTRADICTED:
                if metadata["authority_reason"] == "untrusted_instruction_promotion":
                    findings.append(Finding(category=RiskCategory.POLICY, subtype="untrusted_instruction_promotion", severity=Severity.CRITICAL, confidence=1.0, status=FindingStatus.CONTRADICTED, message="Lower-trust document text was promoted to an instruction or system policy.", detector=self.name, span=claim.span, evidence=claim.evidence, metadata=metadata))
                else:
                    findings.append(Finding(category=RiskCategory.HALLUCINATION, subtype="claim_contradicted", severity=Severity.HIGH, confidence=float(claim.confidence or 0.5), status=FindingStatus.CONTRADICTED, message="A factual claim is contradicted by retrieved evidence.", detector=self.name, span=claim.span, evidence=claim.evidence, metadata=metadata))
            elif claim.status == VerificationStatus.UNSUPPORTED:
                findings.append(Finding(category=RiskCategory.HALLUCINATION, subtype="claim_unsupported", severity=Severity.HIGH if interaction.consequential else Severity.MEDIUM, confidence=float(claim.confidence or 0.5), status=FindingStatus.UNSUPPORTED, message="A checkable factual claim is not supported by a sufficiently complete evidence set.", detector=self.name, span=claim.span, evidence=claim.evidence, metadata=metadata))
            elif claim.status == VerificationStatus.CONFLICTING:
                findings.append(Finding(category=RiskCategory.HALLUCINATION, subtype="claim_conflicting", severity=Severity.HIGH if interaction.consequential else Severity.MEDIUM, confidence=float(claim.confidence or 0.5), status=FindingStatus.CONFLICTING, message="Strong evidence sources materially disagree about this factual claim.", detector=self.name, span=claim.span, evidence=claim.evidence, metadata=metadata))
            else:
                metadata["unknown_reason"] = self._legacy_unknown_reason(claim, assessment)
                findings.append(Finding(category=RiskCategory.HALLUCINATION, subtype="claim_undecidable", severity=Severity.HIGH if interaction.consequential else Severity.LOW, confidence=float(claim.confidence or 0.5), status=FindingStatus.UNDECIDABLE, message="Available evidence is insufficient to determine whether this factual claim is supported.", detector=self.name, span=claim.span, evidence=claim.evidence, metadata=metadata))
        return findings


def build_support_scorer_from_env(nli: NLIScorer | None = None) -> SupportScorer | None:
    backend = os.getenv("CONTROLPLANE_SUPPORT_BACKEND", "none").strip().lower()
    if backend in {"", "none", "off", "0"}:
        return None
    if backend == "nli":
        if nli is None:
            raise RuntimeError("CONTROLPLANE_SUPPORT_BACKEND=nli requires a configured NLI scorer")
        return NLIEntailmentSupportScorer(nli)
    if backend == "minicheck":
        return MiniCheckSupportScorer(
            model_name=os.getenv("CONTROLPLANE_MINICHECK_MODEL", "flan-t5-large"),
            cache_dir=os.getenv("CONTROLPLANE_MINICHECK_CACHE", "./ckpts/minicheck"),
        )
    if backend in {"align", "alignscore"}:
        checkpoint = os.getenv("CONTROLPLANE_ALIGNSCORE_CHECKPOINT")
        if not checkpoint:
            raise RuntimeError("CONTROLPLANE_ALIGNSCORE_CHECKPOINT is required for Align/AlignScore-style verification")
        return AlignScoreSupportScorer(
            checkpoint_path=checkpoint,
            model=os.getenv("CONTROLPLANE_ALIGNSCORE_MODEL", "roberta-base"),
            device=os.getenv("CONTROLPLANE_ALIGNSCORE_DEVICE", "cpu"),
        )
    raise ValueError(f"Unknown CONTROLPLANE_SUPPORT_BACKEND: {backend}")


def build_adaptive_verification_from_env() -> AdaptiveFactVerificationService | None:
    enabled = os.getenv("CONTROLPLANE_ADAPTIVE_VERIFICATION", "1").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    model_name = os.getenv("CONTROLPLANE_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
    device = os.getenv("CONTROLPLANE_NLI_DEVICE") or None
    local_files_only = os.getenv("CONTROLPLANE_NLI_LOCAL_ONLY", "1").strip().lower() not in {"0", "false", "no", "off"}
    nli = LazyTransformersNLIScorer(model_name, device=device, local_files_only=local_files_only)
    retrieval = EvidenceRetrieverConfig(min_score=float(os.getenv("CONTROLPLANE_RETRIEVAL_MIN_SCORE", "0.05")))
    agent = BoundedVerificationAgent(nli, [ContextSearchTool(retrieval_config=retrieval)], AgentVerificationConfig())
    support = build_support_scorer_from_env(nli)
    return AdaptiveFactVerificationService(nli=nli, agent=agent, support_scorer=support, retrieval_config=retrieval)
