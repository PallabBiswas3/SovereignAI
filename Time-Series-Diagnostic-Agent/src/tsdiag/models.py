from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal
import json
import numpy as np


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [json_safe(row) for row in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return repr(value)


@dataclass
class SignalRecord:
    signal: np.ndarray
    fs: float
    timestamps: np.ndarray | None = None
    channels: list[str] | None = None
    shaft_rate: float | None = None
    load: float | None = None
    fault_frequencies: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.signal = np.asarray(self.signal, dtype=float)
        if self.signal.ndim == 1:
            self.signal = self.signal[:, None]
        if self.signal.ndim != 2:
            raise ValueError("signal must be [samples] or [samples, channels]")
        if self.signal.shape[0] < 16:
            raise ValueError("signal must contain at least 16 samples")
        if not np.all(np.isfinite(self.signal)):
            raise ValueError("signal contains NaN or infinite values")
        if not np.isfinite(self.fs) or self.fs <= 0:
            raise ValueError("fs must be a positive finite number")


@dataclass
class Evidence:
    source: str
    statement: str
    score: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)
    evidence_id: str | None = None
    kind: Literal["signal", "statistical", "causal", "model", "physics", "knowledge", "prognostic", "other"] = "other"
    provenance: dict[str, Any] = field(default_factory=dict)
    timestamp: str | int | float | None = None


@dataclass
class AgentResult:
    agent: str
    status: Literal["ok", "warning", "abstain", "unavailable"]
    summary: str
    confidence: float
    evidence: list[Evidence] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class DiagnosticReport:
    decision: Literal["diagnose", "monitor", "abstain"]
    label: str | None
    confidence: float
    agent_results: list[AgentResult]
    evidence: list[Evidence]
    trace: list[str]


@dataclass
class DetectionResult:
    abnormal: bool | None
    score: float | None = None
    onset: str | int | float | None = None
    method: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalizationResult:
    components: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticHypothesis:
    label: str
    score: float
    rationale: str = ""
    alternatives: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    hypothesis: str
    status: Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"]
    reason: str
    evidence_ids: list[str] = field(default_factory=list)
    verifier: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class UncertaintyEstimate:
    value: float | None
    method: str
    calibrated: bool = False
    calibration_dataset: str | None = None
    interval: tuple[float, float] | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Abstention:
    abstained: bool
    reason: str | None = None
    trigger: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunProvenance:
    timestamp_utc: str | None = None
    input_hash: str | None = None
    dataset_id: str | None = None
    protocol_id: str | None = None
    artifact_checksums: dict[str, str] = field(default_factory=dict)
    git_sha: str | None = None
    run_id: str | None = None
    source: str | None = None
    workflow_version: str | None = None
    policy_version: str | None = None
    model_versions: dict[str, str] = field(default_factory=dict)


@dataclass
class PrognosisResult:
    risk: float | None = None
    horizon: str | int | float | None = None
    remaining_useful_life: float | None = None
    uncertainty: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolTraceStep:
    tool: str
    status: Literal["ok", "warning", "error", "skipped"] = "ok"
    inputs_summary: dict[str, Any] = field(default_factory=dict)
    outputs_summary: dict[str, Any] = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)
    duration_seconds: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    children: list["ToolTraceStep"] = field(default_factory=list)


def _validate_unit_interval(name: str, value: float | None) -> None:
    if value is None:
        return
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass
class DiagnosticResult:
    domain: str
    task: str
    decision: Literal["diagnose", "monitor", "abstain"]
    detection: DetectionResult
    localization: LocalizationResult = field(default_factory=LocalizationResult)
    hypotheses: list[DiagnosticHypothesis] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    verification: list[VerificationResult] = field(default_factory=list)
    prognosis: PrognosisResult | None = None
    confidence: float = 0.0
    uncertainty: float | None = None
    uncertainty_estimate: UncertaintyEstimate | None = None
    abstained: bool = False
    abstain_reason: str | None = None
    abstention: Abstention | None = None
    recommended_actions: list[str] = field(default_factory=list)
    tool_trace: list[ToolTraceStep] = field(default_factory=list)
    provenance: RunProvenance | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def validate(self) -> None:
        if not self.domain.strip():
            raise ValueError("domain must be non-empty")
        if not self.task.strip():
            raise ValueError("task must be non-empty")
        _validate_unit_interval("confidence", self.confidence)
        _validate_unit_interval("uncertainty", self.uncertainty)
        _validate_unit_interval("detection.score", self.detection.score)
        if self.uncertainty_estimate is not None:
            _validate_unit_interval("uncertainty_estimate.value", self.uncertainty_estimate.value)
        if self.prognosis is not None:
            _validate_unit_interval("prognosis.risk", self.prognosis.risk)
            _validate_unit_interval("prognosis.uncertainty", self.prognosis.uncertainty)
        for index, hypothesis in enumerate(self.hypotheses):
            _validate_unit_interval(f"hypotheses[{index}].score", hypothesis.score)
        if self.decision == "abstain" and not self.abstained:
            raise ValueError("decision='abstain' requires abstained=True")
        if self.abstained and not self.abstain_reason:
            raise ValueError("abstained results require abstain_reason")
        if not self.abstained and self.abstain_reason:
            raise ValueError("abstain_reason must be empty when abstained=False")
        if self.abstention is not None:
            if self.abstention.abstained != self.abstained:
                raise ValueError("typed abstention must agree with compatibility abstained field")
            if self.abstention.reason != self.abstain_reason:
                raise ValueError("typed abstention reason must agree with abstain_reason")

        evidence_ids = [row.evidence_id for row in self.evidence if row.evidence_id]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_id values must be unique within a diagnostic result")
        known_ids = set(evidence_ids)
        for hypothesis in self.hypotheses:
            unknown = set(hypothesis.evidence_ids) - known_ids
            if unknown:
                raise ValueError(f"hypothesis references unknown evidence ids: {sorted(unknown)}")
        for verification in self.verification:
            unknown = set(verification.evidence_ids) - known_ids
            if unknown:
                raise ValueError(f"verification references unknown evidence ids: {sorted(unknown)}")

        def validate_trace(step: ToolTraceStep) -> None:
            unknown = set(step.evidence_ids) - known_ids
            if unknown:
                raise ValueError(f"tool trace references unknown evidence ids: {sorted(unknown)}")
            for child in step.children:
                validate_trace(child)

        for step in self.tool_trace:
            validate_trace(step)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return json_safe(asdict(self))

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_legacy_report(cls, report: DiagnosticReport, *, domain: str, task: str = "diagnosis") -> "DiagnosticResult":
        evidence = [
            Evidence(
                source=row.source,
                statement=row.statement,
                score=row.score,
                details=dict(row.details),
                evidence_id=row.evidence_id or f"legacy-evidence-{index}",
                kind=row.kind,
                provenance=dict(row.provenance),
                timestamp=row.timestamp,
            )
            for index, row in enumerate(report.evidence)
        ]
        hypotheses: list[DiagnosticHypothesis] = []
        if report.label is not None:
            hypotheses.append(DiagnosticHypothesis(
                label=report.label,
                score=float(np.clip(report.confidence, 0.0, 1.0)),
                rationale="Migrated from legacy DiagnosticReport label.",
                evidence_ids=[row.evidence_id for row in evidence if row.evidence_id],
            ))
        abstained = report.decision == "abstain"
        result = cls(
            domain=domain,
            task=task,
            decision=report.decision,
            detection=DetectionResult(
                abnormal=True if report.decision == "diagnose" else (None if abstained else False),
                score=float(np.clip(report.confidence, 0.0, 1.0)),
                method="legacy_report_adapter",
            ),
            hypotheses=hypotheses,
            evidence=evidence,
            confidence=float(np.clip(report.confidence, 0.0, 1.0)),
            abstained=abstained,
            abstain_reason="Legacy diagnostic report abstained." if abstained else None,
            recommended_actions=[recommendation for agent_result in report.agent_results for recommendation in agent_result.recommendations],
            tool_trace=[ToolTraceStep(tool=str(step), status="ok") for step in report.trace],
            metadata={"legacy_agent_results": [asdict(row) for row in report.agent_results], "migrated_from": "DiagnosticReport"},
        )
        result.validate()
        return result
