from __future__ import annotations

from pathlib import Path

from app.workcells.loader import WorkcellLoadError, WorkcellLoader
from app.workcells.models import (
    WorkcellCatalogEntry,
    WorkcellDefinition,
    WorkcellStatus,
    WorkcellTrustStatus,
    WorkcellValidationIssue,
    WorkcellValidationResult,
)
from app.workcells.validator import WorkcellValidator


class WorkcellRegistry:
    def __init__(self, root: Path, loader: WorkcellLoader, validator: WorkcellValidator) -> None:
        self.root = root.resolve()
        self.loader = loader
        self.validator = validator
        self._definitions: dict[tuple[str, str], WorkcellDefinition] = {}
        self._validations: dict[tuple[str, str], WorkcellValidationResult] = {}
        self._invalid: list[WorkcellValidationResult] = []

    def discover(self) -> None:
        self._definitions.clear()
        self._validations.clear()
        self._invalid.clear()
        if not self.root.exists():
            return
        for directory in sorted((item for item in self.root.iterdir() if item.is_dir()), key=lambda item: item.name):
            try:
                definition = self.loader.load(directory)
                key = (definition.manifest.id, definition.manifest.version)
                if key in self._definitions:
                    result = WorkcellValidationResult(
                        valid=False, status=WorkcellStatus.invalid,
                        workcell_id=definition.manifest.id, version=definition.manifest.version,
                        content_hash=definition.content_hash,
                        issues=[WorkcellValidationIssue(code="DUPLICATE_WORKCELL", message="Duplicate Workcell ID and version")],
                    )
                    self._validations[key] = result
                    self._invalid.append(result)
                    continue
                self._definitions[key] = definition
                self._validations[key] = self.validator.validate(definition)
            except Exception as exc:
                self._invalid.append(WorkcellValidationResult(
                    valid=False, status=WorkcellStatus.invalid,
                    issues=[WorkcellValidationIssue(code="WORKCELL_LOAD_FAILED", message=str(exc))],
                ))

    def list(self) -> list[WorkcellCatalogEntry]:
        entries: list[WorkcellCatalogEntry] = []
        for key, definition in sorted(self._definitions.items()):
            validation = self._validations[key]
            entries.append(WorkcellCatalogEntry(
                id=definition.manifest.id, name=definition.manifest.name,
                version=definition.manifest.version, description=definition.manifest.description,
                task_classes=definition.manifest.task_classes,
                required_tools=definition.manifest.required_tools,
                status=validation.status, trust_status=validation.trust_status,
                content_hash=definition.content_hash, validation=validation,
            ))
        return entries

    def get(self, workcell_id: str, version: str | None = None, *, require_ready: bool = True) -> WorkcellDefinition:
        matches = [(key, value) for key, value in self._definitions.items() if key[0] == workcell_id and (version is None or key[1] == version)]
        if not matches:
            raise KeyError(f"WORKCELL_NOT_FOUND: {workcell_id}")
        matches.sort(key=lambda item: item[0][1], reverse=True)
        key, definition = matches[0]
        validation = self._validations[key]
        if require_ready and not validation.valid:
            raise ValueError(f"WORKCELL_{validation.status.value}: {workcell_id}")
        return definition

    def validation(self, workcell_id: str, version: str | None = None) -> WorkcellValidationResult:
        definition = self.get(workcell_id, version, require_ready=False)
        return self._validations[(definition.manifest.id, definition.manifest.version)]

    def resolve_for_task(self, task_class: str) -> WorkcellDefinition | None:
        candidates = [
            definition for key, definition in self._definitions.items()
            if task_class in definition.manifest.task_classes and self._validations[key].valid
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item.manifest.id, item.manifest.version))
        return candidates[0]
