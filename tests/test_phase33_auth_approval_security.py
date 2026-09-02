from __future__ import annotations

import json
import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from app.api.governance import ApprovalDecision, decide_approval
from app.core.config import get_settings
from app.core.database import Base, HumanApprovalRecord, IdentitySessionRecord, SessionLocal, UserRecord, init_db
from app.identity import ContentIdentityService
from app.identity.models import ClearanceLevel, Permission, Principal, Role
from app.identity.passwords import PasswordHasher
from app.identity.provider import LocalIdentityProvider
from app.main import app


def _principal(user_id: str, roles: list[Role], permissions: list[Permission]) -> Principal:
    return Principal(
        user_id=user_id, email=f"{user_id}@apel.local", display_name=user_id,
        organization_id="apel", department_ids=["maintenance"], workspace_ids=["apel-plant-a"],
        roles=roles, permissions=permissions, clearance=ClearanceLevel.restricted,
    )


def test_requester_cannot_approve_and_argument_mutation_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOVEREIGN_AUTH_MODE", "local")
    get_settings.cache_clear()
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    requester = _principal("requester", [Role.engineer, Role.approver], [Permission.tool_execute, Permission.approval_request, Permission.approval_approve, Permission.task_read])
    approver = _principal("approver", [Role.approver], [Permission.approval_approve, Permission.task_read])
    args = {"path": "approval-tests/safe.md", "content": "original"}
    with Session(engine) as session:
        session.add(UserRecord(
            id="requester", email=requester.email, email_normalized=requester.email,
            display_name="Requester", organization_id="apel",
            department_ids_json='["maintenance"]', workspace_ids_json='["apel-plant-a"]',
            roles_json='["ENGINEER"]', permissions_json='[]', clearance="RESTRICTED",
            password_hash=PasswordHasher().hash("TestSecret!123"), enabled=True,
        ))
        approval = HumanApprovalRecord(
            id=str(uuid4()), tool="write_file", args_json=json.dumps(args), risk="HIGH", status="pending",
            requester_id="requester", organization_id="apel", workspace_id="apel-plant-a",
            action_hash=ContentIdentityService().hash_json({"tool": "write_file", "arguments": args}),
        )
        session.add(approval)
        session.commit()
        with pytest.raises(HTTPException) as denied:
            asyncio.run(decide_approval(approval.id, ApprovalDecision(approve=True), requester, session))
        assert denied.value.detail["code"] == "APPROVER_SEPARATION_REQUIRED"

        approval.args_json = json.dumps({**args, "content": "mutated"})
        session.commit()
        result = asyncio.run(decide_approval(approval.id, ApprovalDecision(approve=True), approver, session))
        assert result["status"] == "blocked"
        assert result["result"]["error"] == "APPROVAL_ARGUMENTS_CHANGED"
    get_settings.cache_clear()


def test_local_http_login_csrf_expiration_and_disabled_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOVEREIGN_AUTH_MODE", "local")
    get_settings.cache_clear()
    init_db()
    suffix = uuid4().hex[:8]
    enabled_id, disabled_id = f"http-user-{suffix}", f"http-disabled-{suffix}"
    email, disabled_email = f"{enabled_id}@apel.local", f"{disabled_id}@apel.local"
    with SessionLocal() as session:
        for user_id, address, enabled in ((enabled_id, email, True), (disabled_id, disabled_email, False)):
            session.add(UserRecord(
                id=user_id, email=address, email_normalized=address,
                display_name="HTTP Test User", organization_id="apel",
                department_ids_json='["maintenance"]', workspace_ids_json='["apel-plant-a"]',
                roles_json='["USER"]', permissions_json='[]', clearance="INTERNAL",
                password_hash=PasswordHasher().hash("Secret!123"), enabled=enabled,
            ))
        session.commit()
    try:
        with TestClient(app) as client:
            invalid = client.post("/api/auth/login", json={"email": f"missing-{suffix}@apel.local", "password": "Secret!123"})
            disabled = client.post("/api/auth/login", json={"email": disabled_email, "password": "Secret!123"})
            assert invalid.status_code == disabled.status_code == 401
            assert invalid.json()["detail"]["code"] == disabled.json()["detail"]["code"] == "INVALID_CREDENTIALS"

            logged_in = client.post("/api/auth/login", json={"email": email, "password": "Secret!123"})
            assert logged_in.status_code == 200
            assert logged_in.json()["principal"]["user_id"] == enabled_id
            assert client.get("/api/auth/me").status_code == 200
            assert client.post("/api/auth/logout").json()["detail"]["code"] == "CSRF_VALIDATION_FAILED"
            csrf = client.cookies.get("sovereign_csrf")
            assert csrf and client.post("/api/auth/logout", headers={"X-CSRF-Token": csrf}).status_code == 200
            assert client.get("/api/auth/me").status_code == 401

            client.post("/api/auth/login", json={"email": email, "password": "Secret!123"})
            token = client.cookies.get("sovereign_session")
            assert token
            with SessionLocal() as session:
                row = session.query(IdentitySessionRecord).filter(IdentitySessionRecord.token_hash == LocalIdentityProvider.token_hash(token)).one()
                row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
                session.commit()
            expired = client.get("/api/auth/me")
            assert expired.status_code == 401
            assert expired.json()["detail"]["code"] == "SESSION_EXPIRED"
    finally:
        with SessionLocal() as session:
            session.query(IdentitySessionRecord).filter(IdentitySessionRecord.user_id.in_([enabled_id, disabled_id])).delete(synchronize_session=False)
            session.query(UserRecord).filter(UserRecord.id.in_([enabled_id, disabled_id])).delete(synchronize_session=False)
            session.commit()
        get_settings.cache_clear()
