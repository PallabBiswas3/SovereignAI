"""Deprecated v0 specialist-agent API.

Use :func:`tsdiag.diagnose` and ``DiagnosticResult`` for new integrations. The
classes remain importable for reproducibility and will not be removed before a
major-version release.
"""

from .base import DiagnosticAgent
from .suite import (
    SignalProcessingAgent,
    BearingDiagnosticAgent,
    StatisticalMonitoringAgent,
    ProbabilisticDiagnosticAgent,
    CausalRootCauseAgent,
    TransferRobustnessAgent,
    LearnedModelAgent,
    MultimodalFusionAgent,
    EvidenceVerificationAgent,
    PrognosticsAgent,
)
from .orchestrator import AgentRegistry, IndustrialDiagnosticOrchestrator

__all__ = [
    "DiagnosticAgent",
    "SignalProcessingAgent",
    "BearingDiagnosticAgent",
    "StatisticalMonitoringAgent",
    "ProbabilisticDiagnosticAgent",
    "CausalRootCauseAgent",
    "TransferRobustnessAgent",
    "LearnedModelAgent",
    "MultimodalFusionAgent",
    "EvidenceVerificationAgent",
    "PrognosticsAgent",
    "AgentRegistry",
    "IndustrialDiagnosticOrchestrator",
]
