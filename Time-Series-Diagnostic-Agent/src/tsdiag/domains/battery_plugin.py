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
            return {"battery_prognosis": result}

        return Workflow((Step("battery_prognosis", prognosis, version="capacity-history-v2"),), version="prognosis-1.0")

    def policy(self, request: DiagnosticRequest):
        if self._is_prognosis(request):
            return PassthroughDecisionPolicy(
                "battery_prognosis", "prognosis-1.0", request,
                version="capacity-prognosis-policy-v1",
            )
        return super().policy(request)
