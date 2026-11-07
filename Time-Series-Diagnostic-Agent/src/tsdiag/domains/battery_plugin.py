from __future__ import annotations

from typing import Any, Mapping

from ..contracts import DiagnosticRequest
from ..execution import Step, Workflow
from .battery_pack_plugin import BatteryPackPlugin
from .battery_prognosis_runner import BatteryPrognosticPipeline
from .policies import PassthroughDecisionPolicy


class BatteryPlugin(BatteryPackPlugin):
    """Task-aware battery plugin: decomposed pack diagnosis plus capacity prognosis."""

    @staticmethod
    def _is_prognosis(request: DiagnosticRequest) -> bool:
        return str(request.task or "").strip().lower() == "prognosis"

    def validate(self, request: DiagnosticRequest) -> Mapping[str, Any]:
        if not self._is_prognosis(request):
            return super().validate(request)
        values = dict(request.inputs)
        missing = [key for key in ("cycle_index", "capacity_ah") if values.get(key) is None]
        if missing:
            raise ValueError(f"battery prognosis requires {', '.join(missing)}")
        return values

    def workflow(self, request: DiagnosticRequest) -> Workflow:
        if not self._is_prognosis(request):
            return super().workflow(request)

        def prognosis(state: dict[str, Any]) -> dict[str, Any]:
            result = BatteryPrognosticPipeline(
                nominal_capacity_ah=float(state.get("nominal_capacity_ah", 2.0)),
                eol_capacity_ah=float(state.get("eol_capacity_ah", 1.4)),
                minimum_observations=int(state.get("minimum_observations", 20)),
                slope_window=int(state.get("slope_window", 20)),
            ).run(
                state["cycle_index"],
                state["capacity_ah"],
                battery_id=state.get("battery_id", "cell"),
            )
            adapter = state.get("capacity_prognostic_model")
            if adapter is None:
                return {"battery_prognosis": result}
            from ..models import UncertaintyEstimate
            report = adapter.predict_diagnostic(state)
            if report.status == "completed" and not result.abstained:
                result.prognosis.remaining_useful_life = report.point_estimate
                result.prognosis.risk = None
                result.prognosis.horizon = "cycles"
                result.prognosis.uncertainty = None
                result.prognosis.details = {"method": report.method, "interval": report.interval,
                    "interval_calibrated": False, "last_observed_cycle": state["cycle_index"][-1],
                    "eol_capacity_ah": state["eol_capacity_ah"]}
                for hypothesis in result.hypotheses:
                    if hypothesis.label == "capacity_fade":
                        hypothesis.details["rul_cycles"] = report.point_estimate
                        hypothesis.details["forecast_method"] = report.method
                result.uncertainty = None
                result.uncertainty_estimate = UncertaintyEstimate(None, report.method, calibrated=False,
                    interval=report.interval, details={"interval_is_model_based": True})
                result.metadata["allow_confidence_complement_uncertainty"] = False
                result.metadata["confidence_interpretation"] = "baseline_capacity_evidence_not_forecast_accuracy"
            return {"battery_prognosis": result, "capacity_update": report.model_dump(mode="json")}

        return Workflow((Step("battery_prognosis", prognosis, version="capacity-history-v3"),), version="prognosis-1.1")

    def policy(self, request: DiagnosticRequest):
        if self._is_prognosis(request):
            return PassthroughDecisionPolicy(
                "battery_prognosis", "prognosis-1.1", request,
                version="capacity-prognosis-policy-v1",
            )
        return super().policy(request)
