from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import warnings

import numpy as np

from .base import DiagnosticAgent
from .suite import (
    BearingDiagnosticAgent,
    CausalRootCauseAgent,
    EvidenceVerificationAgent,
    LearnedModelAgent,
    MultimodalFusionAgent,
    ProbabilisticDiagnosticAgent,
    PrognosticsAgent,
    SignalProcessingAgent,
    StatisticalMonitoringAgent,
    TransferRobustnessAgent,
)
from ..models import AgentResult, DiagnosticReport, Evidence, SignalRecord


@dataclass
class AgentRegistry:
    agents: dict[str, DiagnosticAgent]

    @classmethod
    def default(cls) -> "AgentRegistry":
        instances: Iterable[DiagnosticAgent] = [
            SignalProcessingAgent(),
            BearingDiagnosticAgent(),
            StatisticalMonitoringAgent(),
            ProbabilisticDiagnosticAgent(),
            CausalRootCauseAgent(),
            TransferRobustnessAgent(),
            LearnedModelAgent(),
            MultimodalFusionAgent(),
            EvidenceVerificationAgent(),
            PrognosticsAgent(),
        ]
        return cls({agent.name: agent for agent in instances})


class IndustrialDiagnosticOrchestrator:
    """
    Deprecated bounded specialist-agent orchestrator.

    v0 uses transparent routing rules instead of an LLM so that future LLM/RL
    routing can be compared against a deterministic baseline. New integrations
    should call ``tsdiag.diagnose`` and consume ``DiagnosticResult``.
    """

    def __init__(self, registry: AgentRegistry | None = None):
        warnings.warn(
            "IndustrialDiagnosticOrchestrator is the deprecated v0 API; "
            "use tsdiag.diagnose() for new integrations.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.registry = registry or AgentRegistry.default()

    def _route(self, record: SignalRecord, context: dict) -> list[str]:
        record.validate()
        route = ["signal_processing"]

        if record.fault_frequencies:
            route.append("bearing_fault")

        if context.get("reference_windows") is not None:
            route.extend(["statistical_monitoring", "probabilistic", "transfer_robustness"])

        if record.signal.shape[1] >= 2:
            route.append("causal_root_cause")

        if context.get("predictor") is not None:
            route.append("learned_model")

        if context.get("modal_evidence"):
            route.append("multimodal_fusion")

        if context.get("claims"):
            route.append("evidence_verification")

        if context.get("health_history") is not None:
            route.append("prognostics")

        # Preserve order while removing accidental duplicates.
        seen = set()
        return [name for name in route if not (name in seen or seen.add(name))]

    def run(self, record: SignalRecord, context: dict | None = None) -> DiagnosticReport:
        context = context or {}
        route = self._route(record, context)
        results: list[AgentResult] = []
        trace: list[str] = []

        for name in route:
            agent = self.registry.agents[name]
            result = agent.run(record, context)
            results.append(result)
            trace.append(f"{name}: {result.status} — {result.summary}")

        usable = [r for r in results if r.status not in {"abstain", "unavailable"}]
        warnings = [r for r in usable if r.status == "warning"]

        all_evidence: list[Evidence] = []
        for result in usable:
            all_evidence.extend(result.evidence)

        if not usable:
            return DiagnosticReport("abstain", None, 0.0, results, all_evidence, trace)

        if warnings:
            # Prefer the most confident warning-producing specialist as the headline.
            best = max(warnings, key=lambda r: r.confidence)
            label = best.summary
            confidence = float(np.average([r.confidence for r in warnings]))
            decision = "diagnose"
        else:
            label = "No specialist agent produced a strong fault warning."
            confidence = float(np.average([r.confidence for r in usable]))
            decision = "monitor"

        verifier = next((r for r in results if r.agent == "evidence_verification"), None)
        if verifier and verifier.status == "warning":
            decision = "abstain"
            label = "Candidate diagnosis was contradicted by at least one verification hook."
            confidence = min(confidence, verifier.confidence)
            trace.append("orchestrator: abstain because verification contradicted candidate claim(s)")

        return DiagnosticReport(
            decision=decision,
            label=label,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            agent_results=results,
            evidence=all_evidence,
            trace=trace,
        )
