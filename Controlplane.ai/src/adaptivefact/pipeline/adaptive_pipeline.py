from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from adaptivefact.agents.verifier import BoundedVerificationAgent
from adaptivefact.benchmark.runner import Component, ComponentResult
from adaptivefact.data.schema import (
    ResponseLabel,
    ResponseRecord,
    RiskMetadata,
    RiskModelVersion,
    TimingMetadata,
    VerificationMetadata,
    VerificationRoute,
)
from adaptivefact.pipeline.aggregation import AggregationResult, aggregate_claims
from adaptivefact.pipeline.prioritization import prioritize_unknown_claims
from adaptivefact.pipeline.remediation import EvidenceGroundedRemediator, RemediationConfig
from adaptivefact.routing.router import AdaptiveRouter
from adaptivefact.verification.phase5 import Phase5Pipeline
from adaptivefact.verification.phase6 import Phase6Pipeline


@dataclass
class AdaptivePipelineConfig:
    lightweight_max_nli_claims: int = 3
    deep_max_nli_claims: int = 20
    agent_min_claim_priority: float = 0.75
    agent_max_claims: int = 5
    agent_response_latency_budget_ms: float = 10_000.0
    important_unknown_threshold: float = 0.75
    consequential_metadata_key: str = "consequential"
    auto_remediation_enabled: bool = True
    remediation_min_supported_confidence: float = 0.90
    auto_abstain_nonconsequential: bool = True


class AdaptiveVerificationPipeline:
    """Phase 7/8 dispatcher connecting risk, rules, NLI, and agentic search."""

    def __init__(
        self,
        router: AdaptiveRouter,
        phase5: Phase5Pipeline,
        phase6: Phase6Pipeline,
        *,
        agent: BoundedVerificationAgent | None = None,
        config: AdaptivePipelineConfig | None = None,
    ) -> None:
        self.router = router
        self.phase5 = phase5
        self.phase6 = phase6
        self.agent = agent
        self.config = config or AdaptivePipelineConfig()
        self.remediator = EvidenceGroundedRemediator(
            RemediationConfig(
                enabled=self.config.auto_remediation_enabled,
                min_supported_confidence=self.config.remediation_min_supported_confidence,
                auto_abstain_nonconsequential=self.config.auto_abstain_nonconsequential,
            )
        )

    def process(self, record: ResponseRecord) -> tuple[ResponseRecord, dict]:
        total_started = perf_counter()
        routing = self.router.route(record)
        record.risk_metadata = RiskMetadata(
            risk_score=routing.raw_risk,
            calibrated_risk_score=routing.calibrated_risk,
            model_version=RiskModelVersion.V1,
            latency_ms=routing.latency_ms,
        )

        if routing.route == VerificationRoute.FAST_ACCEPT:
            record.verification_metadata = VerificationMetadata(
                route=routing.route,
                final_decision=ResponseLabel.SUPPORTED,
            )
            total_ms = (perf_counter() - total_started) * 1000.0
            record.timing_metadata = TimingMetadata(
                risk_latency_ms=routing.latency_ms,
                verification_latency_ms=0.0,
                total_latency_ms=total_ms,
                stage_breakdown_ms={"risk_and_routing": routing.latency_ms},
            )
            remediation = self.remediator.remediate(
                record,
                ResponseLabel.SUPPORTED,
                consequential=bool(
                    record.metadata.get(self.config.consequential_metadata_key, False)
                ),
            )
            runtime = {
                "route": routing.route.value,
                "raw_risk": routing.raw_risk,
                "calibrated_risk": routing.calibrated_risk,
                "phase5": {},
                "phase6": {},
                "agent": self._empty_agent_runtime(),
                "aggregation": {
                    "label": ResponseLabel.SUPPORTED.value,
                    "reason": "Calibrated risk was below the fast-accept threshold.",
                },
                "remediation": remediation.to_dict(),
                "total_ms": total_ms,
            }
            record.metadata["policy_controlled_response"] = remediation.response
            record.metadata["adaptive_pipeline"] = runtime
            return record, runtime

        verification_started = perf_counter()
        _, phase5_runtime = self.phase5.process(record)
        consequential = bool(record.metadata.get(self.config.consequential_metadata_key, False))
        ranked_unknown = prioritize_unknown_claims(
            record.atomic_claims,
            routing.calibrated_risk,
            consequential=consequential,
        )

        nli_limit = (
            self.config.lightweight_max_nli_claims
            if routing.route == VerificationRoute.LIGHTWEIGHT
            else self.config.deep_max_nli_claims
        )
        nli_claims = ranked_unknown[: max(0, nli_limit)]
        _, phase6_runtime = self.phase6.process_records(
            [record],
            claim_ids={claim.id for claim in nli_claims},
        )

        agent_results = []
        agent_candidates = []
        response_budget_exhausted = False
        if routing.route == VerificationRoute.AGENTIC and self.agent is not None:
            agent_candidates = prioritize_unknown_claims(
                record.atomic_claims,
                routing.calibrated_risk,
                consequential=consequential,
                min_priority=self.config.agent_min_claim_priority,
                limit=self.config.agent_max_claims,
            )
            agent_started = perf_counter()
            for claim in agent_candidates:
                agent_elapsed_ms = (perf_counter() - agent_started) * 1000.0
                if agent_elapsed_ms >= self.config.agent_response_latency_budget_ms:
                    response_budget_exhausted = True
                    break
                result = self.agent.verify(record, claim)
                self.agent.apply(claim, result)
                agent_results.append(result)

        aggregation = aggregate_claims(
            record.atomic_claims,
            important_unknown_threshold=self.config.important_unknown_threshold,
        )
        remediation = self.remediator.remediate(
            record,
            aggregation.label,
            consequential=consequential,
        )
        verification_ms = (perf_counter() - verification_started) * 1000.0
        total_ms = (perf_counter() - total_started) * 1000.0
        agent_runtime = self._summarize_agent(agent_results)
        agent_runtime["response_latency_budget_ms"] = (
            self.config.agent_response_latency_budget_ms
        )
        agent_runtime["response_budget_exhausted"] = response_budget_exhausted
        agent_runtime["claims_skipped_due_response_budget"] = max(
            0,
            len(agent_candidates) - len(agent_results),
        ) if response_budget_exhausted else 0

        record.verification_metadata = VerificationMetadata(
            route=routing.route,
            claims_checked=len(record.atomic_claims),
            retrieval_call_count=int(phase6_runtime.get("nli_claims", 0)),
            search_call_count=agent_runtime["search_calls"],
            agent_iterations=agent_runtime["iterations"],
            final_decision=aggregation.label,
        )
        record.timing_metadata = TimingMetadata(
            risk_latency_ms=routing.latency_ms,
            verification_latency_ms=verification_ms,
            total_latency_ms=total_ms,
            stage_breakdown_ms={
                "risk_and_routing": routing.latency_ms,
                "claim_extraction": float(phase5_runtime.get("extraction_ms", 0.0)),
                "deterministic_verification": float(phase5_runtime.get("verification_ms", 0.0)),
                "nli": float(phase6_runtime.get("nli_batch_latency_ms", 0.0)),
                "agent": float(agent_runtime["latency_ms"]),
            },
        )

        runtime = {
            "route": routing.route.value,
            "raw_risk": routing.raw_risk,
            "calibrated_risk": routing.calibrated_risk,
            "phase5": phase5_runtime,
            "phase6": phase6_runtime,
            "agent": agent_runtime,
            "aggregation": self._aggregation_dict(aggregation),
            "remediation": remediation.to_dict(),
            "total_ms": total_ms,
        }
        record.metadata["agent_traces"] = [item.model_dump(mode="json") for item in agent_results]
        record.metadata["policy_controlled_response"] = remediation.response
        record.metadata["adaptive_pipeline"] = runtime
        return record, runtime

    @staticmethod
    def _empty_agent_runtime() -> dict:
        return {
            "claims": 0,
            "resolved_supported": 0,
            "resolved_contradicted": 0,
            "remaining_unknown": 0,
            "iterations": 0,
            "search_calls": 0,
            "nli_pairs": 0,
            "cost": 0.0,
            "latency_ms": 0.0,
            "stop_reasons": {},
            "response_latency_budget_ms": 0.0,
            "response_budget_exhausted": False,
            "claims_skipped_due_response_budget": 0,
        }

    @classmethod
    def _summarize_agent(cls, results) -> dict:
        if not results:
            return cls._empty_agent_runtime()
        stop_reasons: dict[str, int] = {}
        for item in results:
            stop_reasons[item.stop_reason] = stop_reasons.get(item.stop_reason, 0) + 1
        return {
            "claims": len(results),
            "resolved_supported": sum(item.status.value == "supported" for item in results),
            "resolved_contradicted": sum(item.status.value == "contradicted" for item in results),
            "remaining_unknown": sum(item.status.value == "unknown" for item in results),
            "iterations": sum(item.iterations for item in results),
            "search_calls": sum(item.search_calls for item in results),
            "nli_pairs": sum(item.nli_pairs for item in results),
            "cost": sum(item.total_cost for item in results),
            "latency_ms": sum(item.latency_ms for item in results),
            "stop_reasons": stop_reasons,
        }

    @staticmethod
    def _aggregation_dict(result: AggregationResult) -> dict:
        return {
            "label": result.label.value,
            "reason": result.reason,
            "status_counts": result.status_counts,
            "important_unknown_claims": result.important_unknown_claims,
        }


class AdaptiveVerificationComponent(Component):
    label = "Adaptive verification (Phase 7/8)"

    def __init__(self, pipeline: AdaptiveVerificationPipeline) -> None:
        self.pipeline = pipeline

    def run(self, record: ResponseRecord) -> ComponentResult:
        processed, runtime = self.pipeline.process(record)
        metadata = processed.verification_metadata
        final = metadata.final_decision if metadata is not None else ResponseLabel.UNKNOWN
        prediction = int(final != ResponseLabel.SUPPORTED)
        calibrated = (
            processed.risk_metadata.calibrated_risk_score
            if processed.risk_metadata is not None
            else 0.5
        )
        contradiction_confidences = [
            float(claim.confidence)
            for claim in processed.atomic_claims
            if claim.status.value == "contradicted" and claim.confidence is not None
        ]
        unresolved_priorities = [
            float(claim.risk_score)
            for claim in processed.atomic_claims
            if claim.status.value == "unknown" and claim.risk_score is not None
        ]
        # Component confidence is P(hallucinated), not generic verifier
        # confidence. A 0.98-supported claim must not become 0.98 risk.
        risk_signals = [float(calibrated or 0.0)]
        if final != ResponseLabel.SUPPORTED:
            risk_signals.extend(contradiction_confidences)
            risk_signals.extend(unresolved_priorities)
        confidence = max(risk_signals, default=0.5)
        agent_runtime = runtime.get("agent", {})
        phase6_runtime = runtime.get("phase6", {})
        return ComponentResult(
            prediction=prediction,
            confidence=min(1.0, confidence),
            cost=float(agent_runtime.get("cost", 0.0)),
            search_calls=int(agent_runtime.get("search_calls", 0)),
            retrieval_calls=int(phase6_runtime.get("nli_claims", 0)),
        )
