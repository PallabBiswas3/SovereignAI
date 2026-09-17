from __future__ import annotations

"""Role-shaped DiagnosticResult views; authorization remains host-owned."""

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class PresentationPolicy:
    ref: str
    include_engineering_detail: bool


@dataclass
class PresentationPolicyRegistry:
    _policies: dict[str, PresentationPolicy] = field(default_factory=dict)

    def register(self, policy: PresentationPolicy, *, replace: bool = False) -> None:
        ref = str(policy.ref).strip()
        if not ref:
            raise ValueError("presentation policy ref must be non-empty")
        if ref in self._policies and not replace:
            return
        self._policies[ref] = policy

    def resolve(self, ref: str) -> PresentationPolicy:
        key = str(ref).strip()
        try:
            return self._policies[key]
        except KeyError as exc:
            raise KeyError(
                f"Unknown presentation policy {key!r}; available: {sorted(self._policies)}"
            ) from exc

    def refs(self) -> tuple[str, ...]:
        return tuple(sorted(self._policies))


presentation_policy_registry = PresentationPolicyRegistry()
presentation_policy_registry.register(PresentationPolicy("operator-v1", False))
presentation_policy_registry.register(PresentationPolicy("engineer-v1", True))


def shape_diagnostic_output(
    diagnostic_result: Mapping[str, Any],
    *,
    policy_ref: str,
    registry: PresentationPolicyRegistry = presentation_policy_registry,
) -> dict[str, Any]:
    """Apply a versioned output-shape policy after host authorization."""

    policy = registry.resolve(policy_ref)
    hypotheses = list(diagnostic_result.get("hypotheses") or [])
    localization = dict(diagnostic_result.get("localization") or {})
    base = {
        "presentation_policy_ref": policy.ref,
        "domain": diagnostic_result.get("domain"),
        "task": diagnostic_result.get("task"),
        "decision": diagnostic_result.get("decision"),
        "abnormal": (diagnostic_result.get("detection") or {}).get("abnormal"),
        "top_hypothesis": hypotheses[0].get("label") if hypotheses else None,
        "recommended_actions": list(diagnostic_result.get("recommended_actions") or []),
    }
    if not policy.include_engineering_detail:
        return base
    return {
        **base,
        "confidence": diagnostic_result.get("confidence"),
        "uncertainty_estimate": diagnostic_result.get("uncertainty_estimate"),
        "localization": {
            "components": list(localization.get("components") or []),
            "channels": list(localization.get("channels") or []),
            "scores": dict(localization.get("scores") or {}),
        },
        "hypotheses": hypotheses,
        "evidence": list(diagnostic_result.get("evidence") or []),
        "verification": list(diagnostic_result.get("verification") or []),
        "prognosis": diagnostic_result.get("prognosis"),
        "provenance": diagnostic_result.get("provenance"),
        "tool_trace": list(diagnostic_result.get("tool_trace") or []),
    }
