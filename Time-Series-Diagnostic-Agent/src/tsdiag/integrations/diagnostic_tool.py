from __future__ import annotations

import json
from typing import Any, Mapping

from ..contracts import DiagnosticRequest, RunContext
from ..pipeline import diagnose
from ..input_contract import Envelope, WORKFLOWS, validate_payload, model_input_profile


def _public_schema():
    schema = Envelope.model_json_schema()
    variants = []
    for domain, task, policy, model, _ in WORKFLOWS.values():
        inputs = model.model_json_schema()
        schema.setdefault("$defs", {}).update(inputs.pop("$defs", {}))
        variants.append({"properties": {
            "domain": {"const": domain}, "task": {"const": task},
            "policy_ref": {"const": policy}, "inputs": inputs,
        }})
    schema["oneOf"] = variants
    return schema


DIAGNOSE_TOOL_MANIFEST = {
    "name": "diagnose_industrial_time_series",
    "description": "Run an evidence-backed diagnostic workflow using the versioned industrial input contract.",
    "input_schema": _public_schema(),
}


_REQUEST_FIELDS = {"contract_version", "domain", "task", "inputs", "policy_ref", "model_refs", "run_context"}
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
    return validate_payload(decoded)


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
    request_data["run_context"]["metadata"]["input_profile"] = model_input_profile(request_data)
    request_data["run_context"]["metadata"]["synthetic"] = request_data["run_context"]["metadata"]["provenance"] == "synthetic"
    # Acquisition metadata is retained in run_context, while tool-specific
    # context uses the established scientific runner names.
    if request_data["domain"] == "bearing":
        values = request_data["inputs"]
        values["operating_condition"] = {"shaft_speed_rpm": values["shaft_speed_rpm"], "load_percent": values["load_percent"]}
    request = DiagnosticRequest(
        domain=str(request_data["domain"]),
        task=request_data.get("task"),
        inputs=dict(request_data["inputs"]),
        policy_ref=request_data.get("policy_ref"),
        model_refs=dict(request_data.get("model_refs") or {}),
        run_context=_run_context(request_data.get("run_context")),
    )
    return diagnose(request).to_dict()
