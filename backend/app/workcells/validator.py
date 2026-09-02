from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from app.governance.action_guard import ActionGuard
from app.workcells.handlers import WorkcellHandlerRegistry
from app.capsules.models import CapsuleSignature
from app.capsules.signing import WorkcellTrustStore
from app.workcells.models import (
    WORKCELL_PLATFORM_VERSION,
    WorkcellDefinition,
    WorkcellStatus,
    WorkcellTrustStatus,
    WorkcellValidationIssue,
    WorkcellValidationResult,
)


class WorkcellValidator:
    def __init__(
        self,
        handlers: WorkcellHandlerRegistry,
        tools_config: Path,
        *,
        platform_version: str = WORKCELL_PLATFORM_VERSION,
        unsigned_workcells_allowed: bool = True,
        trust_store: WorkcellTrustStore | None = None,
    ) -> None:
        self.handlers = handlers
        self.tools_config = tools_config
        self.platform_version = platform_version
        self.unsigned_workcells_allowed = unsigned_workcells_allowed
        self.trust_store = trust_store or WorkcellTrustStore()
        raw = yaml.safe_load(tools_config.read_text(encoding="utf-8")) or {}
        self.global_tools: dict[str, dict[str, Any]] = raw.get("tools", {})
        self.guard = ActionGuard(tools_config)

    @staticmethod
    def _issue(code: str, message: str, path: str | None = None) -> WorkcellValidationIssue:
        return WorkcellValidationIssue(code=code, message=message, path=path)

    @staticmethod
    def _validate_schema(schema: Any, name: str) -> list[WorkcellValidationIssue]:
        issues: list[WorkcellValidationIssue] = []
        if not isinstance(schema, dict) or schema.get("type") != "object":
            issues.append(WorkcellValidator._issue("SCHEMA_INVALID", f"{name} must be a JSON object schema", name))
        elif "properties" in schema and not isinstance(schema["properties"], dict):
            issues.append(WorkcellValidator._issue("SCHEMA_INVALID", f"{name} properties must be an object", name))
        return issues

    def validate(self, definition: WorkcellDefinition) -> WorkcellValidationResult:
        issues: list[WorkcellValidationIssue] = []
        manifest = definition.manifest
        try:
            Version(manifest.version)
        except InvalidVersion:
            issues.append(self._issue("VERSION_INVALID", f"Invalid Workcell version: {manifest.version}"))
        try:
            specifier = SpecifierSet(manifest.platform_version)
            if Version(self.platform_version) not in specifier:
                issues.append(self._issue("PLATFORM_INCOMPATIBLE", f"Platform {self.platform_version} does not satisfy {manifest.platform_version}"))
        except (InvalidSpecifier, InvalidVersion):
            issues.append(self._issue("PLATFORM_VERSION_INVALID", f"Invalid platform version constraint: {manifest.platform_version}"))
        issues.extend(self._validate_schema(definition.input_schema, manifest.input_schema.path))
        issues.extend(self._validate_schema(definition.output_schema, manifest.output_schema.path))
        steps = definition.workflow.steps
        ids = [step.id for step in steps]
        if len(ids) != len(set(ids)):
            issues.append(self._issue("DUPLICATE_STEP", "Workflow step IDs must be unique", manifest.entry_workflow))
        known = set(ids)
        declared_outputs = {output for step in steps for output in step.outputs}
        initial_inputs = set(definition.input_schema.get("properties", {}))
        for step in steps:
            for dependency in step.depends_on:
                if dependency not in known:
                    issues.append(self._issue("MISSING_DEPENDENCY", f"Step {step.id} references missing dependency {dependency}", manifest.entry_workflow))
            if not self.handlers.has(step.handler):
                issues.append(self._issue("HANDLER_NOT_FOUND", f"Registered handler not found: {step.handler}", manifest.entry_workflow))
            for reference in step.inputs.values():
                if reference not in declared_outputs and reference not in initial_inputs:
                    issues.append(self._issue("OUTPUT_REFERENCE_MISSING", f"Step {step.id} references undeclared input/output {reference}", manifest.entry_workflow))
        for artifact in definition.artifacts:
            if not self.handlers.has(artifact.handler):
                issues.append(self._issue("ARTIFACT_HANDLER_NOT_FOUND", f"Registered artifact handler not found: {artifact.handler}"))
            if Path(artifact.filename).name != artifact.filename:
                issues.append(self._issue("ARTIFACT_PATH_INVALID", f"Artifact filename must not contain a path: {artifact.filename}"))
        if definition.workflow.terminal_step not in known:
            issues.append(self._issue("TERMINAL_STEP_MISSING", "Workflow terminal_step does not reference a step", manifest.entry_workflow))
        issues.extend(self._cycle_issues(steps))
        required_evidence = {item.id for item in definition.evidence_requirements}
        for step in steps:
            missing = set(step.evidence_requirements) - required_evidence
            if missing:
                issues.append(self._issue("EVIDENCE_REQUIREMENT_MISSING", f"Step {step.id} references unknown evidence requirements: {sorted(missing)}"))
        requested = list(dict.fromkeys(manifest.required_tools + manifest.optional_tools + list(definition.policy.tools)))
        for name in requested:
            config = self.global_tools.get(name)
            if config is None:
                issues.append(self._issue("TOOL_UNKNOWN", f"Unknown tool requested: {name}"))
                continue
            requested_enabled = definition.policy.tools.get(name, True)
            if requested_enabled and not bool(config.get("enabled", False)):
                issues.append(self._issue("TOOL_NOT_ALLOWED", f"Workcell cannot enable globally disabled tool: {name}"))
        trust_status = WorkcellTrustStatus.unsigned
        if definition.signature:
            try:
                signature = CapsuleSignature.model_validate(definition.signature)
                signer = self.trust_store.get(signature.key_id)
                if signer is None:
                    trust_status = WorkcellTrustStatus.signed_unverified
                    issues.append(self._issue("WORKCELL_UNTRUSTED", f"Signing key is not trusted locally: {signature.key_id}"))
                elif signer.verify(definition.content_hash, signature):
                    trust_status = WorkcellTrustStatus.trusted
                else:
                    trust_status = WorkcellTrustStatus.invalid_signature
                    issues.append(self._issue("WORKCELL_SIGNATURE_INVALID", "Workcell signature does not match its content identity"))
            except Exception as exc:
                trust_status = WorkcellTrustStatus.invalid_signature
                issues.append(self._issue("WORKCELL_SIGNATURE_INVALID", str(exc), "signature.json"))
        if not definition.policy.enabled or not manifest.enabled:
            status = WorkcellStatus.disabled
        elif any(item.code == "PLATFORM_INCOMPATIBLE" for item in issues):
            status = WorkcellStatus.incompatible
        elif trust_status == WorkcellTrustStatus.signed_unverified:
            status = WorkcellStatus.untrusted
        elif issues:
            status = WorkcellStatus.invalid
        elif trust_status == WorkcellTrustStatus.unsigned and not self.unsigned_workcells_allowed:
            status = WorkcellStatus.untrusted
            issues.append(self._issue("WORKCELL_UNTRUSTED", "Unsigned Workcells are disabled by policy"))
        else:
            status = WorkcellStatus.ready
        return WorkcellValidationResult(
            valid=status == WorkcellStatus.ready,
            status=status,
            workcell_id=manifest.id,
            version=manifest.version,
            content_hash=definition.content_hash,
            trust_status=trust_status,
            issues=issues,
        )

    @staticmethod
    def _cycle_issues(steps) -> list[WorkcellValidationIssue]:
        graph = {step.id: list(step.depends_on) for step in steps}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            if any(dependency in graph and visit(dependency) for dependency in graph.get(node, [])):
                return True
            visiting.remove(node)
            visited.add(node)
            return False

        if any(visit(node) for node in graph if node not in visited):
            return [WorkcellValidator._issue("WORKFLOW_CYCLE", "Workcell workflow must be a DAG", "workflow.yaml")]
        return []
