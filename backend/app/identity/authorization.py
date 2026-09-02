from __future__ import annotations

from app.identity.models import (
    AccessFailureCode,
    AuthorizationDecision,
    ClearanceLevel,
    DocumentACL,
    EffectiveAccessScope,
    Permission,
    Principal,
    ResourceScope,
    Role,
)


class AuthorizationService:
    """Small RBAC plus bounded organization/workspace/department attributes."""

    def authorize(
        self,
        principal: Principal,
        action: Permission | str,
        resource: ResourceScope | DocumentACL | None = None,
    ) -> AuthorizationDecision:
        permission = action.value if isinstance(action, Permission) else action
        if not principal.has_permission(permission):
            return AuthorizationDecision(
                allowed=False, reason_code=AccessFailureCode.access_denied.value,
                summary="The authenticated principal lacks the required permission.",
            )
        if resource is None:
            return AuthorizationDecision(allowed=True, reason_code="ACCESS_ALLOWED", summary="Permission granted.")
        return self.authorize_resource(principal, resource)

    def authorize_resource(
        self, principal: Principal, resource: ResourceScope | DocumentACL
    ) -> AuthorizationDecision:
        if resource.organization_id != principal.organization_id and principal.organization_id != "*":
            return self._deny(AccessFailureCode.access_denied, "Organization scope does not match.")
        if principal.clearance < resource.classification:
            return self._deny(AccessFailureCode.insufficient_clearance, "Clearance is insufficient.")
        if resource.workspace_id not in principal.workspace_ids and "*" not in principal.workspace_ids:
            return self._deny(AccessFailureCode.workspace_scope_mismatch, "Workspace scope does not match.")
        explicit_acl = bool(resource.allowed_users or resource.allowed_roles)
        explicit_grant = (
            principal.user_id in resource.allowed_users
            or bool(set(resource.allowed_roles).intersection(principal.roles))
            or resource.owner_id == principal.user_id
        )
        department_allowed = (
            resource.department_id is None
            or resource.department_id in principal.department_ids
            or "*" in principal.department_ids
            or principal.has_permission(Permission.knowledge_read_cross_department)
            or principal.user_id in resource.allowed_users
            or bool(set(resource.allowed_roles).intersection(principal.roles))
            or resource.owner_id == principal.user_id
        )
        if not department_allowed:
            return self._deny(AccessFailureCode.department_scope_mismatch, "Department scope does not match.")
        if explicit_acl and not explicit_grant:
            return self._deny(AccessFailureCode.access_denied, "The explicit resource ACL does not grant access.")
        return AuthorizationDecision(allowed=True, reason_code="ACCESS_ALLOWED", summary="Resource scope grants access.")

    def can_read_document(self, principal: Principal, acl: DocumentACL) -> AuthorizationDecision:
        permission = self.authorize(principal, Permission.knowledge_read)
        return permission if not permission.allowed else self.authorize_resource(principal, acl)

    def can_execute_workcell(self, principal: Principal, workcell_id: str) -> AuthorizationDecision:
        decision = self.authorize(principal, Permission.workcell_execute)
        if not decision.allowed:
            return self._deny(AccessFailureCode.workcell_access_denied, "Workcell execution is not permitted.")
        if workcell_id == "pump-inspection" and not (
            {"maintenance", "engineering"}.intersection(principal.department_ids)
            or Role.manager in principal.roles
        ):
            return self._deny(AccessFailureCode.workcell_access_denied, "Workcell is outside the principal's department scope.")
        return decision

    def can_use_tool(self, principal: Principal, tool_name: str) -> AuthorizationDecision:
        if not principal.has_permission(Permission.tool_execute):
            return self._deny(AccessFailureCode.tool_access_denied, "Tool execution is not permitted.")
        if Role.auditor in principal.roles and Role.admin not in principal.roles:
            return self._deny(AccessFailureCode.tool_access_denied, "Auditor role does not grant tool execution.")
        return AuthorizationDecision(allowed=True, reason_code="ACCESS_ALLOWED", summary="Tool permission granted; system and Workcell policy still apply.")

    def can_read_artifact(self, principal: Principal, scope: ResourceScope) -> AuthorizationDecision:
        decision = self.authorize(principal, Permission.artifact_read, scope)
        if not decision.allowed:
            return self._deny(AccessFailureCode.artifact_access_denied, "Artifact access is not permitted.")
        return decision

    def can_read_capsule(self, principal: Principal, scope: ResourceScope) -> AuthorizationDecision:
        decision = self.authorize(principal, Permission.capsule_read, scope)
        if not decision.allowed:
            return self._deny(AccessFailureCode.capsule_access_denied, "Capsule access is not permitted.")
        return decision

    def can_read_asset(self, principal: Principal, scope: ResourceScope) -> AuthorizationDecision:
        decision = self.authorize(principal, Permission.asset_read, scope)
        if not decision.allowed:
            return self._deny(AccessFailureCode.access_denied, "Asset access is not permitted.")
        return decision

    def can_read_telemetry(self, principal: Principal, scope: ResourceScope) -> AuthorizationDecision:
        decision = self.authorize(principal, Permission.telemetry_read, scope)
        if not decision.allowed:
            return self._deny(AccessFailureCode.access_denied, "Telemetry access is not permitted.")
        return decision

    def can_approve_action(self, principal: Principal, requester_id: str | None) -> AuthorizationDecision:
        if not principal.has_permission(Permission.approval_approve) or Role.approver not in principal.roles:
            return self._deny(AccessFailureCode.access_denied, "Business approval authority is required.")
        if requester_id and requester_id == principal.user_id:
            return self._deny(AccessFailureCode.approver_separation_required, "Requester and approver must be different principals.")
        return AuthorizationDecision(allowed=True, reason_code="ACCESS_ALLOWED", summary="Approval authority and separation are valid.")

    def effective_scope(self, principal: Principal) -> EffectiveAccessScope:
        return EffectiveAccessScope(
            organization_id=principal.organization_id,
            department_ids=principal.department_ids,
            workspace_ids=principal.workspace_ids,
            roles=principal.roles,
            user_id=principal.user_id,
            clearance=principal.clearance,
            cross_department=principal.has_permission(Permission.knowledge_read_cross_department),
        )

    @staticmethod
    def owned_scope(principal: Principal, *, classification: ClearanceLevel = ClearanceLevel.internal) -> ResourceScope:
        return ResourceScope(
            organization_id=principal.organization_id,
            department_id=next(iter(principal.department_ids), None),
            workspace_id=next(iter(principal.workspace_ids), ""),
            classification=classification,
            owner_id=principal.user_id,
        )

    @staticmethod
    def _deny(code: AccessFailureCode, summary: str) -> AuthorizationDecision:
        return AuthorizationDecision(allowed=False, reason_code=code.value, summary=summary)
