from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
import hashlib
import json
import secrets
from uuid import uuid4

import yaml
from sqlalchemy.orm import Session

from app.core.database import IdentitySessionRecord, UserRecord
from app.identity.models import ClearanceLevel, Permission, Principal, Role
from app.identity.passwords import PasswordHasher


class IdentityProvider(ABC):
    @abstractmethod
    def authenticate(self, email: str, password: str) -> UserRecord | None:
        raise NotImplementedError

    @abstractmethod
    def create_session(self, user: UserRecord, lifetime_seconds: int) -> tuple[str, IdentitySessionRecord]:
        raise NotImplementedError

    @abstractmethod
    def resolve_principal(self, raw_token: str) -> Principal | None:
        raise NotImplementedError

    @abstractmethod
    def logout(self, raw_token: str) -> bool:
        raise NotImplementedError


class LocalIdentityProvider(IdentityProvider):
    def __init__(self, session: Session, access_config, password_hasher: PasswordHasher | None = None) -> None:
        self.session = session
        self.password_hasher = password_hasher or PasswordHasher()
        raw = yaml.safe_load(access_config.read_text(encoding="utf-8")) or {}
        self.role_permissions = raw.get("roles", {})

    def authenticate(self, email: str, password: str) -> UserRecord | None:
        user = self.session.query(UserRecord).filter(UserRecord.email_normalized == email.strip().lower()).first()
        if not user or not user.enabled or not self.password_hasher.verify(password, user.password_hash):
            return None
        return user

    def create_session(self, user: UserRecord, lifetime_seconds: int) -> tuple[str, IdentitySessionRecord]:
        raw_token = secrets.token_urlsafe(48)
        now = datetime.now(timezone.utc)
        record = IdentitySessionRecord(
            id=str(uuid4()), user_id=user.id,
            token_hash=self.token_hash(raw_token),
            csrf_token=secrets.token_urlsafe(32),
            created_at=now, expires_at=now + timedelta(seconds=lifetime_seconds),
        )
        self.session.add(record)
        self.session.commit()
        return raw_token, record

    def resolve_principal(self, raw_token: str) -> Principal | None:
        record = self.session.query(IdentitySessionRecord).filter(
            IdentitySessionRecord.token_hash == self.token_hash(raw_token)
        ).first()
        if not record or record.revoked:
            return None
        expires = record.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires <= datetime.now(timezone.utc):
            return None
        user = self.session.get(UserRecord, record.user_id)
        if not user or not user.enabled:
            return None
        return self._principal(user, record.id)

    def logout(self, raw_token: str) -> bool:
        record = self.session.query(IdentitySessionRecord).filter(
            IdentitySessionRecord.token_hash == self.token_hash(raw_token)
        ).first()
        if not record:
            return False
        record.revoked = True
        record.revoked_at = datetime.now(timezone.utc)
        self.session.commit()
        return True

    def session_record(self, raw_token: str) -> IdentitySessionRecord | None:
        return self.session.query(IdentitySessionRecord).filter(
            IdentitySessionRecord.token_hash == self.token_hash(raw_token)
        ).first()

    def principal_for_user(self, user_id: str) -> Principal | None:
        user = self.session.get(UserRecord, user_id)
        if not user or not user.enabled:
            return None
        return self._principal(user, "authorization-recheck")

    def _principal(self, user: UserRecord, session_id: str) -> Principal:
        roles = [Role(value) for value in json.loads(user.roles_json)]
        explicit = set(json.loads(user.permissions_json or "[]"))
        for role in roles:
            explicit.update(self.role_permissions.get(role.value, []))
        known = {value.value for value in Permission}
        return Principal(
            user_id=user.id, email=user.email, display_name=user.display_name,
            organization_id=user.organization_id,
            department_ids=json.loads(user.department_ids_json),
            workspace_ids=json.loads(user.workspace_ids_json),
            roles=roles, clearance=ClearanceLevel.parse(user.clearance),
            permissions=[Permission(value) for value in sorted(explicit) if value in known],
            session_id=session_id, authentication_mode="local",
        )

    @staticmethod
    def token_hash(raw_token: str) -> str:
        return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
