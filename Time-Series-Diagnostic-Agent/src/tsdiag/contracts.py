from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol


class DataKind(str, Enum):
    WAVEFORM = "waveform"
    MULTIVARIATE_SERIES = "multivariate_series"
    SCADA = "scada"
    BATTERY_PACK = "battery_pack"
    RUN_TO_FAILURE = "run_to_failure"
    MULTISENSOR_WAVEFORM = "multisensor_waveform"


class TaskKind(str, Enum):
    FAULT_DIAGNOSIS = "fault_diagnosis"
    ROOT_CAUSE = "root_cause"
    CONDITION_MONITORING = "condition_monitoring"
    ANOMALY_LOCALIZATION = "anomaly_localization"
    PROGNOSIS = "prognosis"
    RUL = "remaining_useful_life"


@dataclass(frozen=True)
class ToolContract:
    name: str
    purpose: str
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    evidence_fields: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()
    implementation: str | None = None


@dataclass(frozen=True)
class DomainPack:
    key: str
    title: str
    data_kind: DataKind
    tasks: tuple[TaskKind, ...]
    required_metadata: tuple[str, ...]
    optional_metadata: tuple[str, ...]
    tools: tuple[ToolContract, ...]
    outputs: tuple[str, ...]
    benchmark_targets: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tools)


@dataclass
class ToolRegistry:
    implementations: dict[str, Callable[..., Any]] = field(default_factory=dict)

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        if not callable(fn):
            raise TypeError(f"Tool implementation for {name!r} must be callable")
        self.implementations[name] = fn

    def get(self, name: str) -> Callable[..., Any] | None:
        return self.implementations.get(name)

    def missing_for(self, pack: DomainPack) -> list[str]:
        return [name for name in pack.tool_names() if name not in self.implementations]


@dataclass(frozen=True)
class RunContext:
    run_id: str | None = None
    source: str | None = None
    dataset_id: str | None = None
    protocol_id: str | None = None
    artifact_checksums: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosticRequest:
    domain: str
    task: str | None
    inputs: Mapping[str, Any]
    policy_ref: str | None = None
    model_refs: Mapping[str, str] = field(default_factory=dict)
    run_context: RunContext | None = None


class DecisionPolicy(Protocol):
    version: str

    def decide(self, *args: Any, **kwargs: Any): ...


class DomainPlugin(Protocol):
    name: str
    workflow_version: str

    def validate(self, request: DiagnosticRequest) -> Mapping[str, Any]: ...
    def workflow(self, request: DiagnosticRequest): ...
    def policy(self, request: DiagnosticRequest) -> DecisionPolicy: ...
