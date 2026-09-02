from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import DepartmentRecord, OrganizationRecord, UserRecord, WorkspaceRecord, get_db
from app.identity.dependencies import get_current_principal, require_permission
from app.identity.models import Permission, Principal


router = APIRouter(prefix="/api", tags=["organization"])


@router.get("/organization")
async def organization(
    principal: Principal = Depends(get_current_principal), db: Session = Depends(get_db)
) -> dict[str, object]:
    organization = db.get(OrganizationRecord, principal.organization_id)
    departments = db.query(DepartmentRecord).filter(DepartmentRecord.organization_id == principal.organization_id).order_by(DepartmentRecord.name).all()
    workspaces = db.query(WorkspaceRecord).filter(WorkspaceRecord.organization_id == principal.organization_id).order_by(WorkspaceRecord.name).all()
    return {
        "organization": ({"id": organization.id, "name": organization.name, "short_name": organization.short_name} if organization else None),
        "metadata": json.loads(organization.metadata_json or "{}") if organization else {},
        "departments": [{"id": item.id, "name": item.name} for item in departments],
        "workspaces": [{"id": item.id, "name": item.name} for item in workspaces],
    }


@router.get("/users/me/permissions")
async def my_permissions(principal: Principal = Depends(get_current_principal)) -> dict[str, object]:
    return {"user_id": principal.user_id, "permissions": [item.value for item in principal.permissions]}


@router.get("/admin/users")
async def list_users(
    principal: Principal = Depends(require_permission(Permission.admin_manage_users)),
    db: Session = Depends(get_db),
) -> list[dict[str, object]]:
    records = db.query(UserRecord).filter(UserRecord.organization_id == principal.organization_id).order_by(UserRecord.display_name).all()
    return [{
        "id": item.id, "email": item.email, "display_name": item.display_name,
        "departments": json.loads(item.department_ids_json), "workspaces": json.loads(item.workspace_ids_json),
        "roles": json.loads(item.roles_json), "clearance": item.clearance, "enabled": item.enabled,
    } for item in records]
