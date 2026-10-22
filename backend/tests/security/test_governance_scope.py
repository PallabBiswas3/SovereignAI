"""Synthetic regressions for approval and audit resource authorization."""
import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.governance import ApprovalDecision, decide_approval, get_approval, get_audit, router
from app.core.config import get_settings
from app.core.database import AgentRunRecord, AuditEventRecord, Base, HumanApprovalRecord, get_db
from app.identity.dependencies import get_current_principal
from app.identity.models import ClearanceLevel, Permission, Principal, Role


def principal(**updates):
    values = dict(user_id="approver", email="synthetic@example.invalid", display_name="Synthetic",
                  organization_id="org-a", workspace_ids=["workspace-a"], department_ids=["engineering"],
                  roles=[Role.approver], clearance=ClearanceLevel.restricted,
                  permissions=[Permission.task_read, Permission.approval_approve, Permission.audit_read])
    return Principal(**(values | updates))


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setenv("SOVEREIGN_AUTH_MODE", "local")
    get_settings.cache_clear()
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(AgentRunRecord(id="run-a", request="synthetic", status="completed", state_json="{}",
                                   organization_id="org-a", workspace_id="workspace-a", owner_id="requester"))
        session.add(HumanApprovalRecord(id="approval-a", run_id="run-a", tool="write_file", args_json="{}",
                                        risk="HIGH", status="pending", requester_id="requester",
                                        organization_id="org-a", workspace_id="workspace-a"))
        session.add(AuditEventRecord(id="audit-a", run_id="run-a", event_type="synthetic", summary="Synthetic",
                                     payload_json="{}", organization_id="org-a"))
        session.commit()
        yield session
    engine.dispose()
    get_settings.cache_clear()


@pytest.mark.parametrize("updates", [
    {"organization_id": "org-b"}, {"workspace_ids": ["workspace-b"]},
    {"clearance": ClearanceLevel.public},
])
@pytest.mark.parametrize("operation", ["read", "approve", "reject", "audit"])
def test_foreign_or_insufficient_scope_is_denied_before_mutation(db, updates, operation):
    from fastapi import HTTPException
    actor = principal(**updates)
    with pytest.raises(HTTPException) as caught:
        if operation == "read":
            asyncio.run(get_approval("approval-a", actor, db))
        elif operation == "audit":
            asyncio.run(get_audit("run-a", actor, db))
        else:
            asyncio.run(decide_approval("approval-a", ApprovalDecision(approve=operation == "approve"), actor, db))
    assert caught.value.status_code == 404
    assert db.get(HumanApprovalRecord, "approval-a").status == "pending"


def test_authorized_reads_and_rejection_still_work(db):
    actor = principal()
    assert asyncio.run(get_approval("approval-a", actor, db))["id"] == "approval-a"
    assert len(asyncio.run(get_audit("run-a", actor, db))["events"]) == 1
    result = asyncio.run(decide_approval("approval-a", ApprovalDecision(approve=False), actor, db))
    assert result["status"] == "rejected"


def test_malformed_saved_task_cannot_be_approved(db):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        asyncio.run(decide_approval("approval-a", ApprovalDecision(approve=True), principal(), db))
    assert caught.value.status_code == 409
    assert caught.value.detail["code"] == "APPROVAL_TASK_STATE_INVALID"
    assert db.get(HumanApprovalRecord, "approval-a").status == "pending"


def test_unscoped_legacy_approval_is_not_accessible(db):
    from fastapi import HTTPException
    db.get(HumanApprovalRecord, "approval-a").workspace_id = None
    db.commit()
    with pytest.raises(HTTPException):
        asyncio.run(get_approval("approval-a", principal(), db))


def test_orphan_and_cross_organization_audit_events_do_not_leak(db):
    from fastapi import HTTPException
    db.add(AuditEventRecord(id="audit-b", run_id="run-a", event_type="synthetic", summary="Foreign",
                            organization_id="org-b"))
    db.commit()
    assert [item["id"] for item in asyncio.run(get_audit("run-a", principal(), db))["events"]] == ["audit-a"]
    with pytest.raises(HTTPException):
        asyncio.run(get_audit("unknown-parent", principal(), db))


def test_permission_dependency_is_enforced_over_http(db):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_principal] = lambda: principal(permissions=[])
    with TestClient(app) as client:
        assert client.get("/api/approvals/approval-a").status_code == 403
        assert client.get("/api/audit/run-a").status_code == 403
        assert client.post("/api/approvals/approval-a", json={"approve": False}).status_code == 403
