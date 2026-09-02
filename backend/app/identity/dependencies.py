from __future__ import annotations

from datetime import datetime, timezone
import json

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.identity.models import AuthFailureCode, ClearanceLevel, Permission, Principal, Role
from app.identity.provider import LocalIdentityProvider


def development_principal() -> Principal:
    return Principal(
        user_id="development-user", email="development@local", display_name="Development User",
        organization_id="*", department_ids=["*"], workspace_ids=["*"],
        roles=[Role.admin, Role.engineer, Role.approver, Role.manager],
        clearance=ClearanceLevel.restricted, permissions=list(Permission),
        authentication_mode="disabled",
    )


def get_current_principal(request: Request, db: Session = Depends(get_db)) -> Principal:
    settings = get_settings()
    if settings.auth_mode.lower() == "disabled":
        principal = development_principal()
        request.state.principal = principal
        return principal
    if settings.auth_mode.lower() != "local":
        raise HTTPException(status_code=503, detail={"code": "IDENTITY_PROVIDER_UNAVAILABLE"})
    token = request.cookies.get(settings.auth_cookie_name)
    if not token:
        raise HTTPException(status_code=401, detail={"code": AuthFailureCode.authentication_required.value})
    provider = LocalIdentityProvider(db, settings.access_config)
    principal = provider.resolve_principal(token)
    if principal is None:
        record = provider.session_record(token)
        code = AuthFailureCode.authentication_required
        if record:
            expires = record.expires_at
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if expires <= datetime.now(timezone.utc):
                code = AuthFailureCode.session_expired
        raise HTTPException(status_code=401, detail={"code": code.value})
    request.state.principal = principal
    return principal


def require_permission(permission: Permission):
    def dependency(principal: Principal = Depends(get_current_principal)) -> Principal:
        if not principal.has_permission(permission):
            raise HTTPException(status_code=403, detail={"code": "ACCESS_DENIED"})
        return principal
    return dependency
