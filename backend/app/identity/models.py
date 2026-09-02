from __future__ import annotations

import hashlib
from enum import Enum, IntEnum
from pydantic import BaseModel, ConfigDict, Field


class StrictIdentityModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClearanceLevel(IntEnum):
    public = 0
    internal = 1
    confidential = 2
    restricted = 3

    @classmethod
    def parse(cls, value: str | "ClearanceLevel") -> "ClearanceLevel":
        if isinstance(value, cls):
            return value
        return cls[str(value).strip().lower()]


class Role(str, Enum):
    user = "USER"
    engineer = "ENGINEER"
    approver = "APPROVER"
    manager = "MANAGER"
    auditor = "AUDITOR"
    admin = "ADMIN"


class Permission(str, Enum):
    knowledge_read = "knowledge.read"
    knowledge_read_cross_department = "knowledge.read.cross_department"
    knowledge_ingest = "knowledge.ingest"
    task_create = "task.create"
    task_read = "task.read"
    workcell_execute = "workcell.execute"
    workcell_manage = "workcell.manage"
    artifact_create = "artifact.create"
    artifact_read = "artifact.read"
    capsule_create = "capsule.create"
    capsule_read = "capsule.read"
    capsule_verify = "capsule.verify"
    tool_execute = "tool.execute"
    approval_request = "approval.request"
    approval_approve = "approval.approve"
    audit_read = "audit.read"
    admin_manage_users = "admin.manage_users"
    asset_read = "asset.read"
    telemetry_read = "telemetry.read"
    maintenance_read = "maintenance.read"
    maintenance_draft_create = "maintenance.draft.create"


class AuthFailureCode(str, Enum):
    authentication_required = "AUTHENTICATION_REQUIRED"
    invalid_credentials = "INVALID_CREDENTIALS"
    session_expired = "SESSION_EXPIRED"
    account_disabled = "ACCOUNT_DISABLED"


class AccessFailureCode(str, Enum):
    access_denied = "ACCESS_DENIED"
    insufficient_clearance = "INSUFFICIENT_CLEARANCE"
    department_scope_mismatch = "DEPARTMENT_SCOPE_MISMATCH"
    workspace_scope_mismatch = "WORKSPACE_SCOPE_MISMATCH"
    document_access_denied = "DOCUMENT_ACCESS_DENIED"
    artifact_access_denied = "ARTIFACT_ACCESS_DENIED"
    capsule_access_denied = "CAPSULE_ACCESS_DENIED"
    workcell_access_denied = "WORKCELL_ACCESS_DENIED"
    tool_access_denied = "TOOL_ACCESS_DENIED"
    approver_separation_required = "APPROVER_SEPARATION_REQUIRED"
    access_scope_required = "ACCESS_SCOPE_REQUIRED"


class Organization(StrictIdentityModel):
    id: str
    name: str
    short_name: str


class Department(StrictIdentityModel):
    id: str
    organization_id: str
    name: str


class Workspace(StrictIdentityModel):
    id: str
    organization_id: str
    name: str


class User(StrictIdentityModel):
    id: str
    email: str
    display_name: str
    organization_id: str
    department_ids: list[str]
    workspace_ids: list[str]
    roles: list[Role]
    clearance: ClearanceLevel
    enabled: bool = True


class Principal(StrictIdentityModel):
    user_id: str
    email: str
    display_name: str
    organization_id: str
    department_ids: list[str] = Field(default_factory=list)
    workspace_ids: list[str] = Field(default_factory=list)
    roles: list[Role] = Field(default_factory=list)
    clearance: ClearanceLevel = ClearanceLevel.internal
    permissions: list[Permission] = Field(default_factory=list)
    session_id: str | None = None
    authentication_mode: str = "local"

    def has_permission(self, permission: Permission | str) -> bool:
        value = permission.value if isinstance(permission, Permission) else permission
        return any(item.value == value for item in self.permissions)


class DocumentACL(StrictIdentityModel):
    organization_id: str
    department_id: str | None = None
    workspace_id: str
    classification: ClearanceLevel = ClearanceLevel.internal
    allowed_roles: list[Role] = Field(default_factory=list)
    allowed_users: list[str] = Field(default_factory=list)
    owner_id: str | None = None


class ResourceScope(DocumentACL):
    resource_id: str | None = None


class AccessContext(StrictIdentityModel):
    principal: Principal
    action: Permission
    resource_type: str
    resource: ResourceScope | None = None
    task_id: str | None = None
    workcell_id: str | None = None
    tool_name: str | None = None


class AuthorizationDecision(StrictIdentityModel):
    allowed: bool
    reason_code: str
    summary: str


class EffectiveAccessScope(StrictIdentityModel):
    organization_id: str
    department_ids: list[str]
    workspace_ids: list[str]
    roles: list[Role]
    user_id: str
    clearance: ClearanceLevel
    cross_department: bool = False

    @property
    def fingerprint(self) -> str:
        values = [
            self.organization_id,
            self.user_id,
            str(int(self.clearance)),
            "1" if self.cross_department else "0",
            *sorted(self.department_ids),
            *sorted(self.workspace_ids),
            *sorted(role.value for role in self.roles),
        ]
        return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()


class SessionInfo(StrictIdentityModel):
    id: str
    user_id: str
    expires_at: str
    revoked: bool = False


class RoleAssignment(StrictIdentityModel):
    user_id: str
    roles: list[Role]
    department_ids: list[str]
    workspace_ids: list[str]
    permissions: list[Permission]
