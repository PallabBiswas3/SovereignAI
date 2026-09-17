from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping

from .contracts import DiagnosticRequest, DomainPlugin


@dataclass
class DomainRegistry:
    _plugins: dict[str, DomainPlugin] = field(default_factory=dict)

    def register(self, plugin: DomainPlugin) -> None:
        name = str(plugin.name).strip().lower()
        if not name:
            raise ValueError("domain plugin name must be non-empty")
        self._plugins[name] = plugin

    def resolve(self, domain: str) -> DomainPlugin:
        key = str(domain).strip().lower()
        try:
            return self._plugins[key]
        except KeyError as exc:
            raise KeyError(f"No domain plugin registered for {key!r}; available: {sorted(self._plugins)}") from exc

    def get(self, domain: str) -> DomainPlugin | None:
        return self._plugins.get(str(domain).strip().lower())

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))


PolicyFactory = Callable[[DiagnosticRequest], Any]


@dataclass(frozen=True)
class PolicyRecord:
    domain: str
    ref: str
    factory: PolicyFactory
    supported_tasks: tuple[str, ...] = ()

    def build(self, request: DiagnosticRequest) -> Any:
        task = str(request.task or "").strip().lower()
        if self.supported_tasks and task not in self.supported_tasks:
            raise ValueError(
                f"Policy {self.ref!r} for {self.domain!r} does not support task {task!r}; "
                f"supported tasks: {sorted(self.supported_tasks)}"
            )
        policy = self.factory(request)
        if not hasattr(policy, "version"):
            raise TypeError(f"Policy {self.ref!r} did not expose a version")
        version = str(policy.version)
        if self.ref != "default" and version != self.ref:
            raise ValueError(
                f"Policy registry ref {self.ref!r} resolved policy version {version!r}; "
                "explicit refs must resolve the concrete registered policy"
            )
        return policy


@dataclass
class PolicyRegistry:
    _records: dict[tuple[str, str], PolicyRecord] = field(default_factory=dict)

    @staticmethod
    def _key(domain: str, ref: str) -> tuple[str, str]:
        return str(domain).strip().lower(), str(ref).strip()

    def register(self, domain: str, ref: str, factory: PolicyFactory, *, supported_tasks: tuple[str, ...] = (), replace: bool = False) -> None:
        key = self._key(domain, ref)
        if not key[1]:
            raise ValueError("policy ref must be non-empty")
        if not callable(factory):
            raise TypeError("policy factory must be callable")
        if key in self._records and not replace:
            return
        self._records[key] = PolicyRecord(
            domain=key[0], ref=key[1], factory=factory,
            supported_tasks=tuple(str(task).strip().lower() for task in supported_tasks),
        )

    def resolve(self, domain: str, ref: str) -> PolicyRecord:
        key = self._key(domain, ref)
        try:
            return self._records[key]
        except KeyError as exc:
            available = sorted(r for (d, r) in self._records if d == key[0])
            raise KeyError(f"Unknown policy {key[1]!r} for {key[0]!r}; available: {available}") from exc

    def create(self, domain: str, ref: str, request: DiagnosticRequest) -> Any:
        return self.resolve(domain, ref).build(request)

    def refs(self, domain: str) -> tuple[str, ...]:
        key = str(domain).strip().lower()
        return tuple(sorted(ref for (d, ref) in self._records if d == key))


ModelValidator = Callable[[Any], None]


def _artifact_checksum(artifact: Any) -> str | None:
    if isinstance(artifact, (bytes, bytearray, memoryview)):
        return sha256(bytes(artifact)).hexdigest()
    if isinstance(artifact, (str, Path)):
        path = Path(artifact)
        if path.is_file():
            digest = sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
    return None


@dataclass(frozen=True)
class ModelRecord:
    ref: str
    version: str
    artifact: Any = None
    validator: ModelValidator | None = None
    checksum: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate_artifact(self) -> None:
        if self.artifact is not None and self.validator is not None:
            self.validator(self.artifact)


@dataclass
class ModelRegistry:
    """Versioned artifacts that can be validated and injected by workflow slot."""

    _records: dict[str, ModelRecord] = field(default_factory=dict)

    def register(
        self,
        ref: str,
        *,
        version: str,
        artifact: Any = None,
        validator: ModelValidator | None = None,
        checksum: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> ModelRecord:
        key = str(ref).strip()
        if not key:
            raise ValueError("model ref must be non-empty")
        if key in self._records and not replace:
            raise KeyError(f"Model ref {key!r} is already registered")
        if validator is not None and not callable(validator):
            raise TypeError("model validator must be callable")
        record = ModelRecord(
            ref=key,
            version=str(version),
            artifact=artifact,
            validator=validator,
            checksum=checksum or _artifact_checksum(artifact),
            metadata=dict(metadata or {}),
        )
        record.validate_artifact()
        self._records[key] = record
        return record

    def resolve(self, ref: str, *, validate: bool = True) -> ModelRecord:
        key = str(ref).strip()
        try:
            record = self._records[key]
        except KeyError as exc:
            raise KeyError(f"Unknown model ref {key!r}; available: {sorted(self._records)}") from exc
        if validate:
            record.validate_artifact()
        return record

    def get(self, ref: str) -> ModelRecord | None:
        return self._records.get(str(ref).strip())

    def refs(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))


domain_registry = DomainRegistry()
policy_registry = PolicyRegistry()
model_registry = ModelRegistry()
