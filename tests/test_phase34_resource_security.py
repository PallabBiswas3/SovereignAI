from __future__ import annotations

import json

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.artifacts import _allowed as artifact_allowed
from app.api.capsules import _assert_access as assert_capsule_access
from app.api.governance import ToolProposal
from app.api.tasks import CreateTaskRequest, _assert_task_access
from app.core.config import get_settings
from app.core.database import AgentRunRecord, ArtifactRecord, EvidenceCapsuleRecord
from app.identity.models import ClearanceLevel, Permission, Principal, Role


def _principal(departments: list[str], *, roles: list[Role] | None = None, permissions: list[Permission] | None = None) -> Principal:
    return Principal(
        user_id="reader", email="reader@apel.local", display_name="Reader",
        organization_id="apel", department_ids=departments, workspace_ids=["apel-plant-a"],
        roles=roles or [Role.user], clearance=ClearanceLevel.restricted,
        permissions=permissions or [Permission.task_read, Permission.artifact_read, Permission.capsule_read],
    )


def test_task_artifact_and_capsule_ids_do_not_bypass_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOVEREIGN_AUTH_MODE", "local")
    get_settings.cache_clear()
    maintenance = _principal(["maintenance"])
    artifact = ArtifactRecord(
        id="artifact-finance", run_id="run-finance", name="secret.md", media_type="text/markdown",
        path="secret.md", size=10, organization_id="apel", owner_id="finance-owner",
        workspace_id="apel-plant-a", department_id="finance", classification="RESTRICTED",
        allowed_roles_json="[]", allowed_users_json="[]",
    )
    capsule = EvidenceCapsuleRecord(
        id="capsule-finance", task_id="run-finance", state="COMPLETE", path="capsule-finance",
        organization_id="apel", owner_id="finance-owner", workspace_id="apel-plant-a",
        department_id="finance", classification="RESTRICTED",
        allowed_roles_json="[]", allowed_users_json="[]",
    )
    task = AgentRunRecord(
        id="run-finance", request="secret", status="completed", state_json="{}",
        organization_id="apel", owner_id="finance-owner", workspace_id="apel-plant-a",
        department_id="finance", classification="RESTRICTED",
    )
    assert not artifact_allowed(maintenance, artifact)
    with pytest.raises(HTTPException) as capsule_denied:
        assert_capsule_access(maintenance, capsule, Permission.capsule_read)
    with pytest.raises(HTTPException) as task_denied:
        _assert_task_access(maintenance, task)
    assert capsule_denied.value.status_code == task_denied.value.status_code == 404
    get_settings.cache_clear()


def test_auditor_can_verify_shared_capsule_but_cannot_execute_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOVEREIGN_AUTH_MODE", "local")
    get_settings.cache_clear()
    auditor = _principal(
        ["management"], roles=[Role.auditor],
        permissions=[Permission.capsule_read, Permission.capsule_verify, Permission.audit_read],
    )
    capsule = EvidenceCapsuleRecord(
        id="capsule-shared", task_id="run-maint", state="COMPLETE", path="capsule-shared",
        organization_id="apel", owner_id="engineer", workspace_id="apel-plant-a",
        department_id="maintenance", classification="CONFIDENTIAL",
        allowed_roles_json=json.dumps([Role.auditor.value]), allowed_users_json="[]",
    )
    assert_capsule_access(auditor, capsule, Permission.capsule_verify)
    from app.identity.authorization import AuthorizationService
    assert not AuthorizationService().can_use_tool(auditor, "write_file").allowed
    get_settings.cache_clear()


def test_authority_fields_are_rejected_from_browser_payloads() -> None:
    with pytest.raises(ValidationError):
        CreateTaskRequest.model_validate({"request": "test", "roles": ["ADMIN"]})
    with pytest.raises(ValidationError):
        ToolProposal.model_validate({"tool": "write_file", "args": {}, "requester_id": "admin"})
