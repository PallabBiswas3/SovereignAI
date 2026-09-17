from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..contracts import DiagnosticRequest
from ..execution import ExecutionTrace, Step, Workflow, WorkflowExecutor
from ..models import (
    DetectionResult,
    DiagnosticHypothesis,
    DiagnosticResult,
    Evidence,
    LocalizationResult,
    PrognosisResult,
    UncertaintyEstimate,
)
from ..result_contract import standardize_result
from .domain_steps import default_domain_tool_registry
from .verification import turbofan_physics_verification


TURBOFAN_POLICY_VERSION = "turbofan-policy-v2"


def _tool(name: str, state: dict[str, Any]) -> dict[str, Any]:
    return default_domain_tool_registry().get("turbofan", name)(state)


@dataclass(frozen=True)
class TurbofanDecisionPolicy:
    version: str = TURBOFAN_POLICY_VERSION

    def decide(self, execution: Mapping[str, Any], trace: ExecutionTrace) -> DiagnosticResult:
        state = dict(execution)
        confidence = float(state["confidence"])
        abnormal = bool(state["abnormal"])
        evidence = Evidence(
            source="health_index",
            statement=(
                f"Health index={state['current_health']:.2f}; "
                f"trend={state['health_slope']:.4g} per cycle."
            ),
            score=confidence,
            details={
                "selected_channels": state["critical_sensors"],
                "rul_cycles": state["rul_cycles"],
            },
            evidence_id="turbofan-health-evidence",
            kind="prognostic",
        )
        trace.require("health_index_estimation").evidence_ids.append(evidence.evidence_id)
        verification = [turbofan_physics_verification(state, evidence.evidence_id)]
        uncertainty = float(state["uncertainty_score"])
        critical_sensors = list(state["critical_sensors"])
        channel_names = list(state["channel_names"])
        sensor_slopes = np.asarray(state["sensor_slopes"], dtype=float)
        failure_threshold = float(state.get("failure_threshold", 3.0))
        detection_score = float(
            np.clip(max(float(state["current_health"]), 0.0) / max(failure_threshold, 1e-12), 0.0, 1.0)
        )

        return standardize_result(DiagnosticResult(
            domain="turbofan",
            task="remaining_useful_life",
            decision="diagnose" if abnormal else "monitor",
            detection=DetectionResult(abnormal, detection_score, method="health_index_trend"),
            localization=LocalizationResult(
                channels=critical_sensors,
                scores={
                    name: float(min(abs(sensor_slopes[channel_names.index(name)]), 1.0))
                    for name in critical_sensors
                },
            ),
            hypotheses=[
                DiagnosticHypothesis("degradation", confidence, evidence_ids=[evidence.evidence_id])
            ] if abnormal else [],
            evidence=[evidence],
            verification=verification,
            prognosis=PrognosisResult(
                remaining_useful_life=state["rul_cycles"],
                horizon="cycles",
                uncertainty=uncertainty,
                details={"interval": state["rul_interval"], "method": state["rul_method"]},
            ),
            confidence=confidence,
            uncertainty=uncertainty,
            uncertainty_estimate=UncertaintyEstimate(
                uncertainty,
                str(state["rul_method"]),
                calibrated=False,
            ),
            recommended_actions=["Track the health trend and inspect critical sensors."] if abnormal else [],
            tool_trace=trace,
            metadata={
                "health_index": state["health_index"],
                "sensor_slopes": dict(zip(channel_names, sensor_slopes.tolist())),
                "workflow_version": TurbofanPlugin.workflow_version,
                "policy_version": self.version,
                "allow_confidence_complement_uncertainty": False,
            },
        ))


class TurbofanPlugin:
    name = "turbofan"
    workflow_version = "2.0"

    def validate(self, request: DiagnosticRequest) -> Mapping[str, Any]:
        values = dict(request.inputs)
        required = ("signal_matrix", "channel_names", "cycle_index")
        missing = [key for key in required if values.get(key) is None]
        if missing:
            raise ValueError(f"turbofan requires {', '.join(missing)}")
        signal = np.asarray(values["signal_matrix"], dtype=float)
        cycles = np.asarray(values["cycle_index"], dtype=float)
        channels = [str(name) for name in values["channel_names"]]
        if signal.ndim != 2:
            raise ValueError("signal_matrix must be a 2-D [cycles, sensors] array")
        if len(cycles) != len(signal):
            raise ValueError("cycle_index must align with signal_matrix rows")
        if len(channels) != signal.shape[1]:
            raise ValueError("channel_names must align with signal_matrix columns")
        values.update(signal_matrix=signal, cycle_index=cycles, channel_names=channels)
        return values

    def workflow(self, request: DiagnosticRequest) -> Workflow:
        step_ids = (
            "sensor_screening",
            "operating_condition_identification",
            "regime_normalization",
            "degradation_smoothing",
            "sequence_windowing",
            "health_index_estimation",
            "rul_prediction",
            "rul_uncertainty",
            "prognostic_explanation",
            "turbofan_decision",
        )
        steps: list[Step] = []
        previous: str | None = None
        for step_id in step_ids:
            def execute(state, name=step_id):
                return _tool(name, state)
            steps.append(Step(step_id, execute, depends_on=() if previous is None else (previous,), version="1.0"))
            previous = step_id
        return Workflow(tuple(steps), version=self.workflow_version)

    def policy(self, request: DiagnosticRequest) -> TurbofanDecisionPolicy:
        return TurbofanDecisionPolicy()


class TurbofanDiagnosticPipeline:
    """Direct runner retained for benchmark callers; uses the decomposed plugin."""

    def run(self, signal_matrix, channel_names, cycle_index, **context):
        request = DiagnosticRequest(
            domain="turbofan",
            task="remaining_useful_life",
            inputs={
                "signal_matrix": signal_matrix,
                "channel_names": channel_names,
                "cycle_index": cycle_index,
                **context,
            },
        )
        plugin = TurbofanPlugin()
        values = dict(plugin.validate(request))
        execution, trace = WorkflowExecutor().run(plugin.workflow(request), values)
        return plugin.policy(request).decide(execution, trace)
