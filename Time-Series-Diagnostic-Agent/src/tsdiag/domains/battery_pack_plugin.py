from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..contracts import DiagnosticRequest
from ..execution import ExecutionTrace, Step, Workflow
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
from .verification import battery_physics_verification


def _tool(name: str, state: dict[str, Any]) -> dict[str, Any]:
    return default_domain_tool_registry().get("battery", name)(state)


@dataclass(frozen=True)
class BatteryPackDecisionPolicy:
    version: str = "battery-pack-policy-v2"

    def decide(self, execution: Mapping[str, Any], trace: ExecutionTrace) -> DiagnosticResult:
        state = dict(execution)
        confidence = float(state["confidence"])
        abnormal = bool(state["abnormal"])
        top = str(state["top_cell"])
        risk = float(state["failure_probability_by_horizon"])
        prognosis_uncertainty = float(state["prognosis_uncertainty"])
        evidence = Evidence(
            "cell_deviation",
            f"Largest cell-to-pack deviation is {top} (score={state['top_score']:.2f}).",
            confidence,
            {"cell_scores": dict(zip(state["cell_ids"], np.asarray(state["cell_fault_scores"]).tolist()))},
            "battery-cell-evidence",
            "statistical",
        )
        trace.require("cell_deviation_features").evidence_ids.append(evidence.evidence_id)
        verification = [battery_physics_verification(state, evidence.evidence_id)]
        return standardize_result(DiagnosticResult(
            domain="battery",
            task="anomaly_localization",
            decision="diagnose" if abnormal else "monitor",
            detection=DetectionResult(abnormal, confidence, method="robust_cell_to_pack_deviation"),
            localization=LocalizationResult(
                components=[top] if abnormal else [],
                channels=[top] if abnormal else [],
                scores={top: confidence} if abnormal else {},
            ),
            hypotheses=[DiagnosticHypothesis("cell_imbalance", confidence, evidence_ids=[evidence.evidence_id])] if abnormal else [],
            evidence=[evidence],
            verification=verification,
            prognosis=PrognosisResult(
                risk=risk,
                horizon=state["latent_health_state"].get("horizon"),
                uncertainty=prognosis_uncertainty,
            ),
            confidence=confidence,
            uncertainty=prognosis_uncertainty,
            uncertainty_estimate=UncertaintyEstimate(prognosis_uncertainty, "battery_prognostic_uncertainty", calibrated=False),
            recommended_actions=[f"Inspect {top} sensing and balance state."] if abnormal else [],
            tool_trace=trace,
            metadata={
                "ranked_cells": state["ranked_cells"],
                "window_metadata": state["window_metadata"],
                "workflow_version": BatteryPackPlugin.workflow_version,
                "policy_version": self.version,
                "allow_confidence_complement_uncertainty": False,
            },
        ))


class BatteryPackPlugin:
    name = "battery"
    workflow_version = "2.0"

    def validate(self, request: DiagnosticRequest) -> Mapping[str, Any]:
        values = dict(request.inputs)
        required = ("cell_voltage", "cell_temperature", "cell_ids", "timestamps")
        missing = [key for key in required if values.get(key) is None]
        if missing:
            raise ValueError(f"battery requires {', '.join(missing)}")
        return values

    def workflow(self, request: DiagnosticRequest) -> Workflow:
        # Use the canonical Battery DomainPack tool names so the declarative
        # workflow, registry, and documented contracts stay in lockstep.
        ids = (
            "battery_data_quality",
            "operating_state_segmentation",
            "cell_deviation_features",
            "battery_temporal_features",
            "cell_spatial_consistency",
            "adaptive_window_builder",
            "battery_spatiotemporal_model",
            "battery_anomaly_localization",
            "battery_failure_prognosis",
            "battery_decision",
        )
        steps = []
        previous = None
        for step_id in ids:
            def execute(state, name=step_id):
                return _tool(name, state)
            steps.append(Step(step_id, execute, depends_on=() if previous is None else (previous,), version="1.0"))
            previous = step_id
        return Workflow(tuple(steps), version=self.workflow_version)

    def policy(self, request: DiagnosticRequest) -> BatteryPackDecisionPolicy:
        return BatteryPackDecisionPolicy()
