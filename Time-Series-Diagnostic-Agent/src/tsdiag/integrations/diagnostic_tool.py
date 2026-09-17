from __future__ import annotations

import json
from typing import Any, Mapping

from ..contracts import DiagnosticRequest, RunContext
from ..pipeline import diagnose


DIAGNOSE_TOOL_MANIFEST = {
    "name": "diagnose_industrial_time_series",
    "description": (
        "Run an evidence-backed diagnostic workflow for one supported industrial "
        "time-series domain and return a validated DiagnosticResult."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["domain", "inputs"],
        "properties": {
            "domain": {
                "type": "string",
                "enum": ["battery", "bearing", "process", "transformer", "turbofan", "wind_scada"],
            },
            "task": {"type": ["string", "null"]},
            "inputs": {"type": "object"},
            "policy_ref": {"type": ["string", "null"]},
            "model_refs": {"type": "object", "additionalProperties": {"type": "string"}},
            "run_context": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "properties": {
                    "run_id": {"type": ["string", "null"]},
                    "source": {"type": ["string", "null"]},
                    "dataset_id": {"type": ["string", "null"]},
                    "protocol_id": {"type": ["string", "null"]},
                    "artifact_checksums": {"type": "object", "additionalProperties": {"type": "string"}},
                    "metadata": {"type": "object"},
                },
            },
        },
    },
}


_REQUEST_FIELDS = {"domain", "task", "inputs", "policy_ref", "model_refs", "run_context"}
_CONTEXT_FIELDS = {"run_id", "source", "dataset_id", "protocol_id", "artifact_checksums", "metadata"}


def _mapping_payload(payload: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload, str):
        decoded = json.loads(payload)
    elif isinstance(payload, Mapping):
        decoded = dict(payload)
    else:
        raise TypeError("diagnostic tool payload must be a JSON string or mapping")
    if not isinstance(decoded, dict):
        raise ValueError("diagnostic tool payload must decode to a JSON object")
    unknown = sorted(set(decoded) - _REQUEST_FIELDS)
    if unknown:
        raise ValueError(f"unsupported diagnostic tool fields: {unknown}")
    if not isinstance(decoded.get("inputs"), Mapping):
        raise ValueError("diagnostic tool payload requires an inputs object")
    if not str(decoded.get("domain") or "").strip():
        raise ValueError("diagnostic tool payload requires a domain")
    return decoded


def _run_context(value: Any) -> RunContext | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("run_context must be an object or null")
    unknown = sorted(set(value) - _CONTEXT_FIELDS)
    if unknown:
        raise ValueError(f"unsupported run_context fields: {unknown}")
    return RunContext(
        run_id=value.get("run_id"),
        source=value.get("source"),
        dataset_id=value.get("dataset_id"),
        protocol_id=value.get("protocol_id"),
        artifact_checksums=dict(value.get("artifact_checksums") or {}),
        metadata=dict(value.get("metadata") or {}),
    )


def diagnose_tool(payload: str | Mapping[str, Any]) -> dict[str, Any]:
    """Execute ``diagnose`` through a JSON-safe, host-neutral tool boundary.

    Authorization remains the responsibility of the hosting platform. This
    adapter performs strict envelope validation and never evaluates code or
    dynamically imports user-supplied callables.
    """
    request_data = _mapping_payload(payload)
    request = DiagnosticRequest(
        domain=str(request_data["domain"]),
        task=request_data.get("task"),
        inputs=dict(request_data["inputs"]),
        policy_ref=request_data.get("policy_ref"),
        model_refs=dict(request_data.get("model_refs") or {}),
        run_context=_run_context(request_data.get("run_context")),
    )
    return diagnose(request).to_dict()
