from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from app.identity import ContentIdentityService
from app.workcells.models import (
    WorkcellArtifactDefinition,
    WorkcellDefinition,
    WorkcellEvaluationDefinition,
    WorkcellEvidenceRequirement,
    WorkcellManifest,
    WorkcellPolicy,
    WorkcellWorkflow,
)


class WorkcellLoadError(ValueError):
    pass


class WorkcellLoader:
    ALLOWED_ROOT_FILES = {
        "manifest.yaml", "workflow.yaml", "evidence.yaml", "rules.yaml",
        "policy.yaml", "signature.json",
    }
    ALLOWED_DIRECTORIES = {"schemas", "prompts", "artifacts", "evaluations", "expected_outputs"}
    ALLOWED_SUFFIXES = {".yaml", ".yml", ".json", ".txt", ".md"}

    def __init__(self, root: Path, *, max_file_bytes: int = 2 * 1024 * 1024) -> None:
        self.root = root.resolve()
        self.max_file_bytes = max_file_bytes
        self.identity = ContentIdentityService()

    def _contained(self, path: Path, parent: Path) -> Path:
        if path.is_symlink():
            raise WorkcellLoadError(f"Symlinks are not allowed: {path.name}")
        resolved = path.resolve()
        resolved_parent = parent.resolve()
        if resolved != resolved_parent and resolved_parent not in resolved.parents:
            raise WorkcellLoadError(f"Path escapes Workcell directory: {path}")
        return resolved

    def _safe_relative(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise WorkcellLoadError(f"Unsafe relative path: {value}")
        return path

    def _read(self, pack_root: Path, relative: str, *, required: bool = True) -> str | None:
        safe_relative = self._safe_relative(relative)
        path = pack_root / safe_relative
        if not path.exists():
            if required:
                raise WorkcellLoadError(f"Required Workcell file is missing: {relative}")
            return None
        resolved = self._contained(path, pack_root)
        if not resolved.is_file():
            raise WorkcellLoadError(f"Expected a regular file: {relative}")
        if resolved.stat().st_size > self.max_file_bytes:
            raise WorkcellLoadError(f"Workcell file exceeds size limit: {relative}")
        return resolved.read_text(encoding="utf-8")

    @staticmethod
    def _yaml(text: str | None, name: str) -> Any:
        try:
            return yaml.safe_load(text or "")
        except yaml.YAMLError as exc:
            raise WorkcellLoadError(f"Invalid YAML in {name}: {exc}") from exc

    @staticmethod
    def _json(text: str | None, name: str) -> Any:
        try:
            return json.loads(text or "")
        except json.JSONDecodeError as exc:
            raise WorkcellLoadError(f"Invalid JSON in {name}: {exc}") from exc

    def _discover_files(self, pack_root: Path) -> list[Path]:
        files: list[Path] = []
        for path in pack_root.rglob("*"):
            relative = path.relative_to(pack_root)
            if any(part.startswith(".") for part in relative.parts):
                raise WorkcellLoadError(f"Hidden Workcell content is not allowed: {relative.as_posix()}")
            self._contained(path, pack_root)
            if path.is_dir():
                if relative.parts[0] not in self.ALLOWED_DIRECTORIES:
                    raise WorkcellLoadError(f"Unknown Workcell directory: {relative.parts[0]}")
                continue
            if len(relative.parts) == 1 and relative.name not in self.ALLOWED_ROOT_FILES:
                raise WorkcellLoadError(f"Unknown Workcell root file: {relative.name}")
            if len(relative.parts) > 1 and relative.parts[0] not in self.ALLOWED_DIRECTORIES:
                raise WorkcellLoadError(f"Workcell file is outside an allowed directory: {relative.as_posix()}")
            if path.suffix.lower() not in self.ALLOWED_SUFFIXES:
                raise WorkcellLoadError(f"Unsupported Workcell file type: {relative.as_posix()}")
            if path.stat().st_size > self.max_file_bytes:
                raise WorkcellLoadError(f"Workcell file exceeds size limit: {relative.as_posix()}")
            files.append(path)
        return files

    def load(self, directory: Path | str) -> WorkcellDefinition:
        candidate = Path(directory)
        pack_root = candidate if candidate.is_absolute() else self.root / candidate
        pack_root = self._contained(pack_root, self.root)
        if not pack_root.is_dir():
            raise WorkcellLoadError(f"Workcell directory does not exist: {pack_root}")
        files = self._discover_files(pack_root)
        manifest_raw = self._yaml(self._read(pack_root, "manifest.yaml"), "manifest.yaml") or {}
        manifest = WorkcellManifest.model_validate(manifest_raw)
        workflow_raw = self._yaml(
            self._read(pack_root, manifest.entry_workflow), manifest.entry_workflow
        ) or {}
        workflow = WorkcellWorkflow.model_validate(workflow_raw)
        input_schema = self._json(
            self._read(pack_root, manifest.input_schema.path), manifest.input_schema.path
        )
        output_schema = self._json(
            self._read(pack_root, manifest.output_schema.path), manifest.output_schema.path
        )
        policy = WorkcellPolicy.model_validate(
            self._yaml(self._read(pack_root, "policy.yaml", required=False), "policy.yaml") or {}
        )
        evidence_raw = self._yaml(
            self._read(pack_root, "evidence.yaml", required=False), "evidence.yaml"
        ) or {}
        rules_raw = self._yaml(self._read(pack_root, "rules.yaml", required=False), "rules.yaml") or {}
        artifacts: list[WorkcellArtifactDefinition] = []
        evaluations: list[WorkcellEvaluationDefinition] = []
        prompts: dict[str, str] = {}
        for path in sorted(files, key=lambda item: item.as_posix()):
            relative = path.relative_to(pack_root).as_posix()
            if relative.startswith("prompts/"):
                prompts[relative] = path.read_text(encoding="utf-8")
            elif relative.startswith("artifacts/") and path.suffix.lower() in {".yaml", ".yml"}:
                raw = self._yaml(path.read_text(encoding="utf-8"), relative) or {}
                artifacts.append(WorkcellArtifactDefinition.model_validate(raw))
            elif relative.startswith("evaluations/") and path.suffix.lower() in {".yaml", ".yml"}:
                raw = self._yaml(path.read_text(encoding="utf-8"), relative) or {}
                cases = raw.get("cases", []) if isinstance(raw, dict) else raw
                evaluations.extend(WorkcellEvaluationDefinition.model_validate(item) for item in cases)
        identity_files = [path for path in files if path.relative_to(pack_root).as_posix() != "signature.json"]
        file_hashes = self.identity.directory_manifest(pack_root, identity_files)
        signature_text = self._read(pack_root, "signature.json", required=False)
        signature = self._json(signature_text, "signature.json") if signature_text else None
        return WorkcellDefinition(
            root=str(pack_root), manifest=manifest, workflow=workflow,
            input_schema=input_schema, output_schema=output_schema, policy=policy,
            evidence_requirements=[
                WorkcellEvidenceRequirement.model_validate(item)
                for item in evidence_raw.get("requirements", [])
            ],
            rules=list(rules_raw.get("rules", [])), artifacts=artifacts, evaluations=evaluations,
            prompts=prompts,
            prompt_hashes={name: self.identity.hash_bytes(value.encode("utf-8")) for name, value in prompts.items()},
            content_hash=self.identity.hash_directory_manifest(file_hashes), files=file_hashes,
            signature=signature,
        )
