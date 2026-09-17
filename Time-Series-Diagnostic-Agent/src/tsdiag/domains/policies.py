from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..contracts import DiagnosticRequest
from ..execution import ExecutionTrace
from ..models import DiagnosticResult, UncertaintyEstimate
from ..result_contract import standardize_result


def _model_version(request: DiagnosticRequest) -> str | None:
    values = tuple(str(value) for value in request.model_refs.values() if value)
    return values[0] if len(values) == 1 else (";".join(values) if values else None)


def _stamp_passthrough_result(
    result: DiagnosticResult,
    request: DiagnosticRequest,
    trace: ExecutionTrace,
    *,
    workflow_version: str,
    policy_version: str,
) -> DiagnosticResult:
    """Attach an atomic task runner below the shared workflow trace."""
    inner_trace = list(result.tool_trace)
    if trace.steps and inner_trace:
        trace.steps[0].children.extend(inner_trace)
    result.tool_trace = trace
    result.metadata["workflow_version"] = workflow_version
    result.metadata["policy_version"] = policy_version
    result.metadata["model_version"] = _model_version(request)
    result.metadata["model_refs"] = dict(request.model_refs)
    result.metadata["atomic_task_adapter"] = True

    if result.uncertainty is not None and np.isclose(
        float(result.uncertainty), 1.0 - float(result.confidence), atol=1e-12
    ):
        result.uncertainty = None
        result.uncertainty_estimate = UncertaintyEstimate(
            value=None,
            method="not_calibrated_legacy_confidence_only",
            calibrated=False,
        )
        result.metadata["allow_confidence_complement_uncertainty"] = False
    return standardize_result(result)


@dataclass(frozen=True)
class PassthroughDecisionPolicy:
    """Policy for intentionally atomic workflows such as battery prognosis."""

    result_key: str
    workflow_version: str
    request: DiagnosticRequest
    version: str

    def decide(self, execution: Mapping[str, object], trace: ExecutionTrace) -> DiagnosticResult:
        result = execution[self.result_key]
        if not isinstance(result, DiagnosticResult):
            raise TypeError(f"{self.result_key!r} did not produce DiagnosticResult")
        return _stamp_passthrough_result(
            result,
            self.request,
            trace,
            workflow_version=self.workflow_version,
            policy_version=self.version,
        )
