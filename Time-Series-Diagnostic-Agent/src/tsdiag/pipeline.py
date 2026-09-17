from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from .contracts import DiagnosticRequest, RunContext
from .domains import DOMAIN_PACKS
from .execution import DomainInputSchema, InputField, StepExecutionError, WorkflowExecutor
from .models import DetectionResult, DiagnosticResult, RunProvenance, ToolTraceStep
from .registry import domain_registry, model_registry, policy_registry
from .result_contract import standardize_result

PIPELINE_VERSION = "1.1.0"
DEFAULT_TRANSFORMER_ADTFM_REF = "transformer-ad-tfm-at-sgah-v1"
DEFAULT_TRANSFORMER_ADTFM_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "transformer_sgah_ad_tfm"
    / "ad_tfm_at_checkpoint.pt"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_default_transformer_model() -> None:
    if model_registry.get(DEFAULT_TRANSFORMER_ADTFM_REF) is not None:
        return
    checkpoint_value = os.environ.get("TSDIAG_TRANSFORMER_ADTFM_CHECKPOINT")
    checkpoint = Path(checkpoint_value) if checkpoint_value else DEFAULT_TRANSFORMER_ADTFM_CHECKPOINT
    if not checkpoint.is_file():
        return
    from .learned.ad_tfm_at import load_ad_tfm_classifier
    model_registry.register(
        DEFAULT_TRANSFORMER_ADTFM_REF,
        version="ad-tfm-at-sgah-v1",
        artifact=load_ad_tfm_classifier(checkpoint),
        checksum=_file_sha256(checkpoint),
        metadata={"kind": "AD-TFM-AT", "dataset": "SGAH", "input": "raw_waveform"},
    )

DOMAIN_INPUT_SCHEMAS = {
    "bearing": DomainInputSchema("bearing", (InputField("signal"), InputField("sampling_rate_hz"), InputField("fault_frequencies", False))),
    "process": DomainInputSchema("process", (InputField("signal_matrix"), InputField("normal_reference"), InputField("channel_names"), InputField("sampling_rate_hz", False), InputField("trained_fault_classifier", False), InputField("monitoring_method", False))),
    "wind_scada": DomainInputSchema("wind_scada", (
        InputField("train_matrix", description="Healthy/reference SCADA samples used to fit the normal-behavior model."),
        InputField("prediction_matrix", description="SCADA samples to diagnose."),
        InputField("channel_names"),
        InputField("train_timestamps", False),
        InputField("prediction_timestamps", False),
    )),
    "battery": DomainInputSchema("battery", (InputField("cell_voltage"), InputField("cell_temperature"), InputField("cell_ids"), InputField("timestamps"))),
    "turbofan": DomainInputSchema("turbofan", (InputField("signal_matrix"), InputField("channel_names"), InputField("cycle_index"))),
    "transformer": DomainInputSchema("transformer", (InputField("signal_matrix"), InputField("sampling_rate_hz"), InputField("sensor_positions"), InputField("raw_waveform_model", False), InputField("trained_image_model", False))),
}


def get_input_schema(domain: str) -> DomainInputSchema:
    try:
        return DOMAIN_INPUT_SCHEMAS[domain]
    except KeyError as exc:
        raise KeyError(f"Unknown domain {domain!r}. Available: {sorted(DOMAIN_INPUT_SCHEMAS)}") from exc


def _abstain(domain: str, task: str, reason: str, trace: list[ToolTraceStep] | None = None) -> DiagnosticResult:
    return standardize_result(DiagnosticResult(
        domain=domain,
        task=task,
        decision="abstain",
        detection=DetectionResult(abnormal=None, method="pipeline_validation"),
        confidence=0.0,
        uncertainty=None,
        abstained=True,
        abstain_reason=reason,
        tool_trace=trace or [ToolTraceStep("input_validation", status="warning", details={"reason": reason})],
        metadata={"pipeline_version": PIPELINE_VERSION, "allow_confidence_complement_uncertainty": False},
    ), pipeline_version=PIPELINE_VERSION)


@lru_cache(maxsize=1)
def _ensure_plugins() -> None:
    from .domains.battery_pack_plugin import BatteryPackDecisionPolicy
    from .domains.battery_plugin import BatteryPlugin
    from .domains.bearing_plugin import BearingDecisionPolicy, BearingPlugin
    from .domains.policies import PassthroughDecisionPolicy
    from .domains.process_plugin import ProcessDecisionPolicy, ProcessPlugin
    from .domains.transformer_plugin import TransformerDecisionPolicy, TransformerPlugin
    from .domains.turbofan_plugin import TURBOFAN_POLICY_VERSION, TurbofanDecisionPolicy, TurbofanPlugin
    from .domains.wind_scada_plugin import WIND_POLICY_VERSION, WindScadaDecisionPolicy, WindScadaPlugin

    plugins = (BearingPlugin(), ProcessPlugin(), WindScadaPlugin(), BatteryPlugin(), TurbofanPlugin(), TransformerPlugin())
    for plugin in plugins:
        domain_registry.register(plugin)
        policy_registry.register(plugin.name, "default", lambda request, p=plugin: p.policy(request), replace=True)

    policy_registry.register(
        "bearing", "bearing-policy-v2",
        lambda request: BearingDecisionPolicy(
            minimum_confidence=float(request.inputs.get("minimum_confidence", 0.45)),
            minimum_harmonics=int(request.inputs.get("minimum_harmonics", 2)),
        ),
        supported_tasks=("fault_diagnosis", "condition_monitoring"), replace=True,
    )
    policy_registry.register(
        "process", "process-policy-v3", lambda request: ProcessDecisionPolicy(request),
        supported_tasks=("root_cause", "fault_diagnosis", "condition_monitoring"), replace=True,
    )
    policy_registry.register(
        "process", "process-policy-v2", lambda request: ProcessDecisionPolicy(request, version="process-policy-v2"),
        supported_tasks=("root_cause", "fault_diagnosis", "condition_monitoring"), replace=True,
    )
    policy_registry.register(
        "wind_scada", "1.0", lambda request: WindScadaDecisionPolicy(version="1.0"),
        supported_tasks=("condition_monitoring",), replace=True,
    )
    policy_registry.register(
        "wind_scada", WIND_POLICY_VERSION,
        lambda request: WindScadaDecisionPolicy(version=WIND_POLICY_VERSION),
        supported_tasks=("condition_monitoring",), replace=True,
    )
    if model_registry.get("wind-nbm-regime-v1") is None:
        model_registry.register(
            "wind-nbm-regime-v1",
            version="wind-nbm-regime-v1",
            metadata={"kind": "built-in", "purpose": "regime-aware Wind SCADA normal-behavior model"},
        )
    policy_registry.register(
        "battery", "battery-pack-policy-v2", lambda request: BatteryPackDecisionPolicy(),
        supported_tasks=("anomaly_localization", "fault_diagnosis"), replace=True,
    )
    policy_registry.register(
        "battery", "capacity-prognosis-policy-v1",
        lambda request: PassthroughDecisionPolicy(
            "battery_prognosis", "prognosis-1.0", request,
            version="capacity-prognosis-policy-v1",
        ),
        supported_tasks=("prognosis",), replace=True,
    )

    policy_registry.register(
        "turbofan", TURBOFAN_POLICY_VERSION,
        lambda request: TurbofanDecisionPolicy(version=TURBOFAN_POLICY_VERSION),
        supported_tasks=("remaining_useful_life", "prognosis", "condition_monitoring"), replace=True,
    )
    policy_registry.register(
        "turbofan", "compat-1.0",
        lambda request: TurbofanDecisionPolicy(version="compat-1.0"),
        supported_tasks=("remaining_useful_life", "prognosis", "condition_monitoring"), replace=True,
    )
    policy_registry.register(
        "transformer", "transformer-policy-v2", lambda request: TransformerDecisionPolicy(request),
        supported_tasks=("fault_diagnosis", "condition_monitoring"), replace=True,
    )


def _hash_update(hasher, value: Any) -> None:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(b"ndarray:"); hasher.update(str(array.dtype).encode()); hasher.update(str(array.shape).encode()); hasher.update(array.tobytes())
    elif isinstance(value, Mapping):
        hasher.update(b"mapping{")
        for key in sorted(value, key=lambda row: str(row)):
            hasher.update(str(key).encode()); _hash_update(hasher, value[key])
        hasher.update(b"}")
    elif isinstance(value, (list, tuple)):
        hasher.update(b"sequence[")
        for row in value: _hash_update(hasher, row)
        hasher.update(b"]")
    elif callable(value):
        hasher.update(f"callable:{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', type(value).__qualname__)}".encode())
    else:
        hasher.update(repr(value).encode())


def _input_hash(values: Mapping[str, Any]) -> str:
    hasher = hashlib.sha256(); _hash_update(hasher, values); return hasher.hexdigest()


@lru_cache(maxsize=1)
def _git_sha() -> str | None:
    env_sha = os.environ.get("GITHUB_SHA")
    if env_sha: return env_sha
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=2)
        return completed.stdout.strip() or None
    except Exception:
        return None


def _resolve_models(request: DiagnosticRequest) -> tuple[DiagnosticRequest, dict[str, str], dict[str, str]]:
    model_refs = dict(request.model_refs)
    values = dict(request.inputs)
    compatible_default_transformer = (
        request.domain == "transformer"
        and "raw_waveform_model" not in values
        and "trained_image_model" not in values
    )
    if compatible_default_transformer:
        signal = np.asarray(values.get("signal_matrix"))
        positions = tuple(str(name) for name in values.get("sensor_positions", ()))
        expected_positions = ("Ua", "Ub", "Uc", "Ia", "Ib", "Ic")
        if signal.shape == (100, 6) and positions == expected_positions:
            _ensure_default_transformer_model()
    if compatible_default_transformer and model_registry.get(DEFAULT_TRANSFORMER_ADTFM_REF) is not None:
        signal = np.asarray(values.get("signal_matrix"))
        positions = tuple(str(name) for name in values.get("sensor_positions", ()))
        if signal.shape == (100, 6) and positions == ("Ua", "Ub", "Uc", "Ia", "Ib", "Ic"):
            model_refs.setdefault("raw_waveform_model", DEFAULT_TRANSFORMER_ADTFM_REF)
    if not model_refs:
        return request, {}, {}
    versions: dict[str, str] = {}; checksums: dict[str, str] = {}
    for slot, ref in model_refs.items():
        record = model_registry.resolve(ref, validate=True); versions[slot] = record.version
        if record.checksum: checksums[slot] = record.checksum
        if record.artifact is not None: values[slot] = record.artifact
    return DiagnosticRequest(domain=request.domain, task=request.task, inputs=values, policy_ref=request.policy_ref,
                             model_refs=model_refs, run_context=request.run_context), versions, checksums


def _provenance(request: DiagnosticRequest, result: DiagnosticResult, model_versions: dict[str, str], model_checksums: dict[str, str]) -> RunProvenance:
    context = request.run_context; checksums = dict(model_checksums)
    if context is not None: checksums.update({str(k): str(v) for k, v in context.artifact_checksums.items()})
    return RunProvenance(
        timestamp_utc=datetime.now(timezone.utc).isoformat(), input_hash=_input_hash(request.inputs),
        dataset_id=context.dataset_id if context else None, protocol_id=context.protocol_id if context else None,
        artifact_checksums=checksums, git_sha=_git_sha(), run_id=context.run_id if context else None,
        source=context.source if context else None, workflow_version=result.metadata.get("workflow_version"),
        policy_version=result.metadata.get("policy_version"), model_versions=dict(model_versions),
    )


def _diagnose_request(request: DiagnosticRequest) -> DiagnosticResult:
    _ensure_plugins(); task = str(request.task or "diagnosis")
    try:
        effective_request, model_versions, model_checksums = _resolve_models(request)
        plugin = domain_registry.resolve(effective_request.domain)
        policy = policy_registry.create(effective_request.domain, effective_request.policy_ref, effective_request) if effective_request.policy_ref else plugin.policy(effective_request)
        validated = dict(plugin.validate(effective_request))
        workflow = plugin.workflow(effective_request)
        execution, trace = WorkflowExecutor().run(workflow, validated)
        result = policy.decide(execution, trace)
        if effective_request.task is not None: result.task = effective_request.task
        # The workflow object is the authority for workflow provenance. Policies
        # may add metadata, but must not stamp a conflicting workflow version.
        result.metadata["workflow_version"] = workflow.version
        result.metadata.setdefault("pipeline_version", PIPELINE_VERSION)
        if effective_request.run_context is not None:
            result.metadata.setdefault("run_context", {
                "run_id": effective_request.run_context.run_id, "source": effective_request.run_context.source,
                "dataset_id": effective_request.run_context.dataset_id, "protocol_id": effective_request.run_context.protocol_id,
                "metadata": dict(effective_request.run_context.metadata),
            })
        if effective_request.model_refs:
            result.metadata.setdefault("model_refs", dict(effective_request.model_refs)); result.metadata.setdefault("resolved_model_versions", model_versions)
        if effective_request.policy_ref:
            result.metadata["policy_ref"] = effective_request.policy_ref
            if str(result.metadata.get("policy_version")) != str(effective_request.policy_ref):
                raise ValueError(f"requested policy_ref={effective_request.policy_ref!r} but executed policy_version={result.metadata.get('policy_version')!r}")
        result.provenance = _provenance(request, result, model_versions, model_checksums)
        return standardize_result(result, pipeline_version=PIPELINE_VERSION)
    except StepExecutionError as exc:
        return _abstain(request.domain, task, f"Pipeline step failed: {exc}", exc.trace)
    except (TypeError, ValueError, KeyError) as exc:
        return _abstain(request.domain, task, f"Input validation failed: {exc}")
    except Exception as exc:
        return _abstain(request.domain, task, f"Pipeline execution failed in {request.domain}: {exc}")


class DiagnosticPipeline:
    version = PIPELINE_VERSION

    def run(self, domain: str, *, task: str | None = None, metadata: dict[str, Any] | None = None, **inputs) -> DiagnosticResult:
        domain = str(domain).strip().lower()
        if domain not in DOMAIN_PACKS: raise KeyError(f"Unknown domain {domain!r}. Available: {sorted(DOMAIN_PACKS)}")
        values = dict(metadata or {}); values.update(inputs)
        if domain == "wind_scada":
            from .domains.wind_scada_plugin import normalize_wind_inputs
            try:
                values = normalize_wind_inputs(values)
            except ValueError as exc:
                return _abstain(domain, str(task or DOMAIN_PACKS[domain].tasks[0].value), f"Input validation failed: {exc}")
        default_task = str(task or DOMAIN_PACKS[domain].tasks[0].value); supported_tasks = {row.value for row in DOMAIN_PACKS[domain].tasks}
        if task is not None and task not in supported_tasks:
            return _abstain(domain, default_task, f"Unsupported task {task!r}; available tasks: {sorted(supported_tasks)}")
        missing = [key for key in DOMAIN_PACKS[domain].required_metadata if values.get(key) is None]
        if domain == "battery" and default_task == "prognosis" and values.get("cycle_index") is not None: missing = []
        if missing: return _abstain(domain, default_task, f"Missing required metadata: {', '.join(missing)}")
        schema_values = dict(values)
        if domain == "bearing" and schema_values.get("signal") is None: schema_values["signal"] = schema_values.get("signal_matrix")
        if not (domain == "battery" and default_task == "prognosis"):
            try: get_input_schema(domain).validate(schema_values)
            except (TypeError, ValueError) as exc: return _abstain(domain, default_task, f"Input validation failed: {exc}")
        return _diagnose_request(DiagnosticRequest(domain=domain, task=default_task, inputs=values))


def diagnose(request_or_domain: DiagnosticRequest | str, *, task: str | None = None, metadata: dict[str, Any] | None = None, **inputs) -> DiagnosticResult:
    if isinstance(request_or_domain, DiagnosticRequest):
        if task is not None or metadata is not None or inputs: raise TypeError("task/metadata/inputs cannot be combined with DiagnosticRequest")
        return _diagnose_request(request_or_domain)
    return DiagnosticPipeline().run(str(request_or_domain), task=task, metadata=metadata, **inputs)
