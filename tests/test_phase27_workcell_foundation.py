from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.identity import ContentIdentityService
from app.workcells.executor import WorkcellExecutionError, WorkcellExecutor
from app.workcells.handlers import WorkcellHandlerContext, WorkcellHandlerRegistry
from app.workcells.loader import WorkcellLoadError, WorkcellLoader
from app.workcells.models import WorkcellStatus
from app.workcells.registry import WorkcellRegistry
from app.workcells.validator import WorkcellValidator
from app.capsules.signing import Ed25519CapsuleSigner, WorkcellTrustStore


TOOLS = """tools:
  read_file: {risk: LOW, enabled: true}
  knowledge_search: {risk: LOW, enabled: true}
  delete_file: {risk: HIGH, enabled: false}
"""


def make_pack(
    root: Path,
    *,
    name: str = "pack",
    manifest_extra: str = "",
    workflow: str | None = None,
    policy: str = "tools: {}\n",
) -> Path:
    pack = root / name
    (pack / "schemas").mkdir(parents=True)
    (pack / "prompts").mkdir()
    (pack / "manifest.yaml").write_text(
        """id: test-workcell
name: Test Workcell
version: 1.0.0
description: Bounded test pack
platform_version: ">=2.0"
task_classes: [test_task]
supported_execution_modes: [STANDARD]
required_tools: [read_file]
optional_tools: []
risk_class: internal
entry_workflow: workflow.yaml
""" + manifest_extra,
        encoding="utf-8",
    )
    (pack / "workflow.yaml").write_text(
        workflow or """version: 1.0.0
terminal_step: second
steps:
  - id: first
    handler: first_handler
    outputs: [value]
  - id: second
    handler: second_handler
    depends_on: [first]
    inputs: {value: value}
    outputs: [result]
""",
        encoding="utf-8",
    )
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "properties": {}}
    (pack / "schemas" / "input.schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (pack / "schemas" / "output.schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (pack / "policy.yaml").write_text(policy, encoding="utf-8")
    (pack / "prompts" / "summary.txt").write_text("Summarize only cited evidence.", encoding="utf-8")
    return pack


def registries(tmp_path: Path):
    tools = tmp_path / "tools.yaml"
    tools.write_text(TOOLS, encoding="utf-8")
    handlers = WorkcellHandlerRegistry()

    async def first(context, step, inputs):
        return {"value": 7}

    async def second(context, step, inputs):
        return {"result": inputs["value"] * 2}

    handlers.register("first_handler", first)
    handlers.register("second_handler", second)
    loader = WorkcellLoader(tmp_path)
    validator = WorkcellValidator(handlers, tools)
    return handlers, loader, validator


def test_valid_manifest_registry_and_deterministic_identity(tmp_path: Path):
    make_pack(tmp_path)
    _, loader, validator = registries(tmp_path)
    registry = WorkcellRegistry(tmp_path, loader, validator)
    registry.discover()
    entries = registry.list()
    assert len(entries) == 1
    assert entries[0].status == WorkcellStatus.ready
    first = registry.get("test-workcell")
    second = loader.load(tmp_path / "pack")
    assert first.content_hash == second.content_hash
    assert first.prompt_hashes["prompts/summary.txt"]
    assert registry.resolve_for_task("test_task").manifest.id == "test-workcell"


def test_manifest_unknown_field_and_invalid_version_are_rejected(tmp_path: Path):
    pack = make_pack(tmp_path, manifest_extra="dangerous_import: os.system\n")
    _, loader, _ = registries(tmp_path)
    with pytest.raises(Exception, match="dangerous_import"):
        loader.load(pack)
    pack = make_pack(tmp_path, manifest_extra="", name="version")
    text = (pack / "manifest.yaml").read_text(encoding="utf-8").replace("version: 1.0.0", "version: definitely-not-semver")
    (pack / "manifest.yaml").write_text(text, encoding="utf-8")
    _, loader, validator = registries(tmp_path)
    result = validator.validate(loader.load(pack))
    assert not result.valid
    assert any(issue.code == "VERSION_INVALID" for issue in result.issues)


def test_missing_handler_dependency_cycle_and_disabled_tool(tmp_path: Path):
    workflow = """version: 1.0.0
terminal_step: a
steps:
  - id: a
    handler: missing_handler
    depends_on: [b]
  - id: b
    handler: first_handler
    depends_on: [a]
"""
    pack = make_pack(tmp_path, workflow=workflow, policy="tools:\n  delete_file: true\n")
    _, loader, validator = registries(tmp_path)
    result = validator.validate(loader.load(pack))
    codes = {issue.code for issue in result.issues}
    assert {"HANDLER_NOT_FOUND", "WORKFLOW_CYCLE", "TOOL_NOT_ALLOWED"} <= codes


def test_missing_dependency_is_rejected(tmp_path: Path):
    workflow = """version: 1.0.0
terminal_step: a
steps:
  - id: a
    handler: first_handler
    depends_on: [missing]
"""
    pack = make_pack(tmp_path, workflow=workflow)
    _, loader, validator = registries(tmp_path)
    result = validator.validate(loader.load(pack))
    assert any(issue.code == "MISSING_DEPENDENCY" for issue in result.issues)


def test_undeclared_step_output_reference_is_rejected(tmp_path: Path):
    workflow = """version: 1.0.0
terminal_step: a
steps:
  - id: a
    handler: first_handler
    inputs: {value: never_declared}
"""
    pack = make_pack(tmp_path, workflow=workflow)
    _, loader, validator = registries(tmp_path)
    result = validator.validate(loader.load(pack))
    assert any(issue.code == "OUTPUT_REFERENCE_MISSING" for issue in result.issues)


def test_loader_rejects_path_traversal_and_unknown_files(tmp_path: Path):
    pack = make_pack(tmp_path)
    manifest = (pack / "manifest.yaml").read_text(encoding="utf-8").replace("workflow.yaml", "../escape.yaml")
    (pack / "manifest.yaml").write_text(manifest, encoding="utf-8")
    _, loader, _ = registries(tmp_path)
    with pytest.raises(WorkcellLoadError, match="Unsafe relative path"):
        loader.load(pack)
    pack = make_pack(tmp_path, name="unknown")
    (pack / "arbitrary.py").write_text("raise SystemExit", encoding="utf-8")
    with pytest.raises(WorkcellLoadError, match="Unknown Workcell root file"):
        loader.load(pack)


def test_loader_rejects_symlink_escape_when_supported(tmp_path: Path):
    pack = make_pack(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = pack / "prompts" / "escape.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is not available on this platform")
    _, loader, _ = registries(tmp_path)
    with pytest.raises(WorkcellLoadError, match="Symlinks are not allowed"):
        loader.load(pack)


def test_valid_dag_executes_handlers_in_dependency_order(tmp_path: Path):
    pack = make_pack(tmp_path)
    handlers, loader, validator = registries(tmp_path)
    definition = loader.load(pack)
    assert validator.validate(definition).valid
    events: list[str] = []

    async def emit(name, payload):
        events.append(name)

    state = asyncio.run(
        WorkcellExecutor(handlers).execute(
            WorkcellHandlerContext(task_id="task-1", request="test", definition=definition, inputs={}),
            event_callback=emit,
        )
    )
    assert state.completed_steps == ["first", "second"]
    assert state.step_outputs["second"]["result"] == 14
    assert events[-1] == "workcell_completed"


def test_step_failure_stops_safely(tmp_path: Path):
    pack = make_pack(tmp_path)
    handlers, loader, _ = registries(tmp_path)

    async def broken(context, step, inputs):
        raise RuntimeError("bounded failure")

    handlers._handlers["first_handler"] = broken
    with pytest.raises(WorkcellExecutionError, match="WORKCELL_EXECUTION_FAILED"):
        asyncio.run(
            WorkcellExecutor(handlers).execute(
                WorkcellHandlerContext(task_id="task-1", request="test", definition=loader.load(pack), inputs={})
            )
        )


def test_content_identity_canonical_json_and_changed_file(tmp_path: Path):
    identity = ContentIdentityService()
    assert identity.hash_json({"b": 2, "a": 1}) == identity.hash_json({"a": 1, "b": 2})
    path = tmp_path / "value.txt"
    path.write_text("one", encoding="utf-8")
    first = identity.hash_file(path)
    path.write_text("two", encoding="utf-8")
    assert identity.hash_file(path) != first
    assert identity.hash_directory_manifest({"b": "2", "a": "1"}) == identity.hash_directory_manifest({"a": "1", "b": "2"})


def test_duplicate_workcell_id_and_version_are_invalid(tmp_path: Path):
    make_pack(tmp_path, name="one")
    make_pack(tmp_path, name="two")
    _, loader, validator = registries(tmp_path)
    registry = WorkcellRegistry(tmp_path, loader, validator)
    registry.discover()
    entry = registry.list()[0]
    assert entry.status == WorkcellStatus.invalid
    assert any(issue.code == "DUPLICATE_WORKCELL" for issue in entry.validation.issues)


def test_workcell_ed25519_trust_and_modified_detection(tmp_path: Path):
    pack = make_pack(tmp_path)
    tools = tmp_path / "tools.yaml"
    handlers, loader, _ = registries(tmp_path)
    unsigned = loader.load(pack)
    signer = Ed25519CapsuleSigner.generate_for_testing("workcell-test")
    (pack / "signature.json").write_text(signer.sign(unsigned.content_hash).model_dump_json(indent=2), encoding="utf-8")
    trust = WorkcellTrustStore()
    trust.add_ed25519("workcell-test", signer.public_bytes())
    signed = loader.load(pack)
    result = WorkcellValidator(handlers, tools, trust_store=trust).validate(signed)
    assert result.valid
    assert result.trust_status.value == "TRUSTED"
    (pack / "prompts" / "summary.txt").write_text("Modified template", encoding="utf-8")
    modified = WorkcellValidator(handlers, tools, trust_store=trust).validate(loader.load(pack))
    assert not modified.valid
    assert modified.trust_status.value == "INVALID_SIGNATURE"
    assert any(issue.code == "WORKCELL_SIGNATURE_INVALID" for issue in modified.issues)


def test_unsigned_workcell_development_and_strict_policy(tmp_path: Path):
    pack = make_pack(tmp_path)
    handlers, loader, _ = registries(tmp_path)
    tools = tmp_path / "tools.yaml"
    definition = loader.load(pack)
    assert WorkcellValidator(handlers, tools, unsigned_workcells_allowed=True).validate(definition).valid
    strict = WorkcellValidator(handlers, tools, unsigned_workcells_allowed=False).validate(definition)
    assert strict.status == WorkcellStatus.untrusted
    assert not strict.valid
