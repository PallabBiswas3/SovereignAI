from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.database import Base, UserRecord
from app.identity.authorization import AuthorizationService
from app.identity.models import ClearanceLevel, DocumentACL, Permission, Principal, Role
from app.identity.passwords import PasswordHasher
from app.identity.provider import LocalIdentityProvider


ACCESS_CONFIG = Path(__file__).resolve().parents[1] / "config" / "access.yaml"


def database() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def add_user(
    db: Session,
    *,
    user_id: str = "maint-1",
    email: str = "maint@apel.local",
    password: str = "Maintenance!42",
    roles: list[str] | None = None,
    enabled: bool = True,
) -> UserRecord:
    record = UserRecord(
        id=user_id, email=email, email_normalized=email.lower(), display_name="Maintenance User",
        organization_id="apel", department_ids_json=json.dumps(["maintenance"]),
        workspace_ids_json=json.dumps(["plant-a"]), roles_json=json.dumps(roles or ["ENGINEER"]),
        permissions_json="[]", clearance="CONFIDENTIAL",
        password_hash=PasswordHasher().hash(password), enabled=enabled,
    )
    db.add(record); db.commit()
    return record


def principal(*, roles: list[Role], permissions: list[Permission], departments=None) -> Principal:
    return Principal(
        user_id="user-1", email="user@apel.local", display_name="User",
        organization_id="apel", department_ids=departments or ["maintenance"],
        workspace_ids=["plant-a"], roles=roles,
        clearance=ClearanceLevel.confidential, permissions=permissions,
    )


def test_password_hash_is_salted_and_securely_verified() -> None:
    hasher = PasswordHasher()
    first = hasher.hash("Maintenance!42")
    second = hasher.hash("Maintenance!42")
    assert first != second
    assert "Maintenance!42" not in first
    assert hasher.verify("Maintenance!42", first)
    assert not hasher.verify("wrong-password", first)


def test_local_authentication_session_resolution_expiration_and_logout() -> None:
    db = database()
    user = add_user(db)
    provider = LocalIdentityProvider(db, ACCESS_CONFIG)
    assert provider.authenticate(user.email, "wrong-password") is None
    authenticated = provider.authenticate(user.email, "Maintenance!42")
    assert authenticated is not None
    token, session = provider.create_session(authenticated, 3600)
    assert token not in session.token_hash
    resolved = provider.resolve_principal(token)
    assert resolved and resolved.user_id == user.id
    assert resolved.has_permission(Permission.workcell_execute)
    session.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    db.commit()
    assert provider.resolve_principal(token) is None
    token2, _ = provider.create_session(authenticated, 3600)
    assert provider.logout(token2)
    assert provider.resolve_principal(token2) is None


def test_disabled_user_cannot_authenticate() -> None:
    db = database()
    user = add_user(db, enabled=False)
    assert LocalIdentityProvider(db, ACCESS_CONFIG).authenticate(user.email, "Maintenance!42") is None


def test_contextual_document_authorization_and_clearance() -> None:
    service = AuthorizationService()
    engineer = principal(roles=[Role.engineer], permissions=[Permission.knowledge_read])
    maintenance = DocumentACL(
        organization_id="apel", department_id="maintenance", workspace_id="plant-a",
        classification=ClearanceLevel.confidential,
    )
    finance = DocumentACL(
        organization_id="apel", department_id="finance", workspace_id="plant-a",
        classification=ClearanceLevel.restricted,
    )
    assert service.can_read_document(engineer, maintenance).allowed
    denied = service.can_read_document(engineer, finance)
    assert not denied.allowed
    assert denied.reason_code == "INSUFFICIENT_CLEARANCE"


def test_rbac_distinguishes_auditor_admin_and_business_approver() -> None:
    service = AuthorizationService()
    auditor = principal(
        roles=[Role.auditor], permissions=[Permission.audit_read, Permission.capsule_verify]
    )
    assert service.authorize(auditor, Permission.audit_read).allowed
    assert not service.can_use_tool(auditor, "read_file").allowed
    admin = principal(roles=[Role.admin], permissions=[Permission.admin_manage_users])
    assert service.authorize(admin, Permission.admin_manage_users).allowed
    assert not service.can_approve_action(admin, "requester").allowed
    manager = principal(
        roles=[Role.manager, Role.approver], permissions=[Permission.approval_approve]
    )
    assert service.can_approve_action(manager, "different-requester").allowed
    same = service.can_approve_action(manager, manager.user_id)
    assert not same.allowed
    assert same.reason_code == "APPROVER_SEPARATION_REQUIRED"
