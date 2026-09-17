from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import (
    AreaRecord, AssetRecord, DepartmentRecord, OrganizationRecord, PlantRecord,
    UserRecord, WorkspaceRecord, get_db,
)
from app.identity.dependencies import get_current_principal, require_permission
from app.identity.models import Permission, Principal
from app.core.config import get_settings
from app.organizations.importer import OrganizationImportError, OrganizationPackImporter
from app.organizations.loader import OrganizationPackError, OrganizationPackLoader


router = APIRouter(prefix="/api", tags=["organization"])
PACK_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{1,99}$")


class OrganizationPackImportRequest(BaseModel):
    confirm_organization_id: str
    dry_run: bool = True
    rotate_local_passwords: bool = False


def _organization_pack_path(name: str) -> Path:
    if not PACK_NAME.fullmatch(name):
        raise HTTPException(status_code=400, detail={"code": "INVALID_PACK_NAME"})
    root = get_settings().organizations_root.resolve()
    path = (root / name).resolve()
    if root not in path.parents:
        raise HTTPException(status_code=400, detail={"code": "INVALID_PACK_PATH"})
    return path


def _load_own_pack(name: str, principal: Principal):
    try:
        pack = OrganizationPackLoader(_organization_pack_path(name)).load()
    except OrganizationPackError as exc:
        raise HTTPException(
            status_code=400, detail={"code": "ORGANIZATION_PACK_INVALID", "message": str(exc)},
        ) from exc
    if pack.organization_id != principal.organization_id:
        raise HTTPException(
            status_code=403,
            detail={"code": "CROSS_ORGANIZATION_IMPORT_REQUIRES_CLI"},
        )
    return pack


@router.get("/organization")
async def organization(
    principal: Principal = Depends(get_current_principal), db: Session = Depends(get_db)
) -> dict[str, object]:
    organization = db.get(OrganizationRecord, principal.organization_id)
    departments = db.query(DepartmentRecord).filter(DepartmentRecord.organization_id == principal.organization_id).order_by(DepartmentRecord.name).all()
    workspaces = db.query(WorkspaceRecord).filter(WorkspaceRecord.organization_id == principal.organization_id).order_by(WorkspaceRecord.name).all()
    plants = db.query(PlantRecord).filter(PlantRecord.organization_id == principal.organization_id).order_by(PlantRecord.name).all()
    areas = db.query(AreaRecord).filter(AreaRecord.organization_id == principal.organization_id).order_by(AreaRecord.name).all()
    assets = db.query(AssetRecord).filter(AssetRecord.organization_id == principal.organization_id).order_by(AssetRecord.id).limit(200).all()
    metadata = json.loads(organization.metadata_json or "{}") if organization else {}
    if plants:
        metadata["plant"] = {
            "id": plants[0].id,
            "name": plants[0].name,
            "areas": [item.name for item in areas if item.plant_id == plants[0].id],
        }
    metadata["assets"] = [{
        "id": item.id,
        "area": next((area.name for area in areas if area.id == item.area_id), item.area_id),
        "service": item.canonical_name,
    } for item in assets]
    return {
        "organization": ({"id": organization.id, "name": organization.name, "short_name": organization.short_name} if organization else None),
        "metadata": metadata,
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


@router.get("/admin/organization-packs")
async def list_organization_packs(
    principal: Principal = Depends(require_permission(Permission.admin_manage_users)),
    db: Session = Depends(get_db),
) -> list[dict[str, object]]:
    settings = get_settings()
    results: list[dict[str, object]] = []
    for path in sorted(
        (item for item in settings.organizations_root.iterdir() if item.is_dir()),
        key=lambda item: item.name,
    ):
        try:
            pack = OrganizationPackLoader(path).load()
            if pack.organization_id != principal.organization_id:
                continue
            validation = OrganizationPackImporter(
                db, settings=settings, principal=principal,
            ).validate(pack)
            results.append({
                "name": path.name,
                "organization_id": pack.organization_id,
                "fingerprint": pack.pack_fingerprint,
                "valid": validation.valid,
                "counts": validation.counts,
                "issues": [item.model_dump(mode="json") for item in validation.issues],
                "required_secret_environment_variables": validation.required_secret_environment_variables,
            })
        except OrganizationPackError:
            continue
    return results


@router.post("/admin/organization-packs/{pack_name}/validate")
async def validate_organization_pack(
    pack_name: str,
    principal: Principal = Depends(require_permission(Permission.admin_manage_users)),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    pack = _load_own_pack(pack_name, principal)
    return OrganizationPackImporter(
        db, settings=get_settings(), principal=principal,
    ).validate(pack).model_dump(mode="json")


@router.post("/admin/organization-packs/{pack_name}/import")
async def import_organization_pack(
    pack_name: str, payload: OrganizationPackImportRequest,
    principal: Principal = Depends(require_permission(Permission.admin_manage_users)),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    pack = _load_own_pack(pack_name, principal)
    if payload.confirm_organization_id != pack.organization_id:
        raise HTTPException(
            status_code=409, detail={"code": "ORGANIZATION_CONFIRMATION_MISMATCH"},
        )
    try:
        report = OrganizationPackImporter(
            db, settings=get_settings(), principal=principal,
        ).import_pack(
            pack, dry_run=payload.dry_run,
            rotate_local_passwords=payload.rotate_local_passwords,
        )
    except OrganizationImportError as exc:
        raise HTTPException(
            status_code=400, detail={"code": "ORGANIZATION_IMPORT_REJECTED", "message": str(exc)},
        ) from exc
    return report.model_dump(mode="json")
