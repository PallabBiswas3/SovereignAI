from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.audit.logger import AuditLogger
from app.core.config import get_settings
from app.core.database import get_db
from app.identity.dependencies import get_current_principal
from app.identity.models import AuthFailureCode, Principal
from app.identity.provider import LocalIdentityProvider


router = APIRouter(prefix="/api/auth", tags=["authentication"])


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=512)


def _public_principal(principal: Principal) -> dict[str, object]:
    return principal.model_dump(mode="json", exclude={"session_id"})


@router.post("/login")
async def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)) -> dict[str, object]:
    settings = get_settings()
    if settings.auth_mode.lower() != "local":
        raise HTTPException(status_code=409, detail={"code": "LOCAL_AUTH_DISABLED"})
    provider = LocalIdentityProvider(db, settings.access_config)
    user = provider.authenticate(payload.email, payload.password)
    audit_id = f"auth:{uuid4()}"
    if user is None:
        AuditLogger(db).log(audit_id, "LOGIN_FAILED", "Local login failed.", {})
        raise HTTPException(status_code=401, detail={"code": AuthFailureCode.invalid_credentials.value})
    raw_token, session = provider.create_session(user, settings.auth_session_seconds)
    principal = provider.resolve_principal(raw_token)
    assert principal is not None
    response.set_cookie(
        settings.auth_cookie_name, raw_token, max_age=settings.auth_session_seconds,
        httponly=True, secure=settings.auth_cookie_secure, samesite="strict", path="/",
    )
    response.set_cookie(
        settings.auth_csrf_cookie_name, session.csrf_token, max_age=settings.auth_session_seconds,
        httponly=False, secure=settings.auth_cookie_secure, samesite="strict", path="/",
    )
    AuditLogger(db).log(audit_id, "LOGIN_SUCCESS", "Local login succeeded.", {"user_id": user.id})
    return {"principal": _public_principal(principal), "expires_at": session.expires_at.isoformat()}


@router.post("/logout")
async def logout(
    request: Request, response: Response,
    principal: Principal = Depends(get_current_principal),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    settings = get_settings()
    token = request.cookies.get(settings.auth_cookie_name)
    if token and settings.auth_mode.lower() == "local":
        LocalIdentityProvider(db, settings.access_config).logout(token)
    response.delete_cookie(settings.auth_cookie_name, path="/")
    response.delete_cookie(settings.auth_csrf_cookie_name, path="/")
    AuditLogger(db).log(f"auth:{uuid4()}", "LOGOUT", "Local session logged out.", {"user_id": principal.user_id})
    return {"status": "logged_out"}


@router.get("/me")
async def me(principal: Principal = Depends(get_current_principal)) -> dict[str, object]:
    return {"principal": _public_principal(principal)}
