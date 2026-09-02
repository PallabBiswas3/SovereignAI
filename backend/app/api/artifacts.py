from __future__ import annotations

from pathlib import Path
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.artifacts.service import ArtifactService
from app.core.config import get_settings
from app.core.database import ArtifactRecord, get_db
from app.audit.logger import AuditLogger
from app.identity.authorization import AuthorizationService
from app.identity.dependencies import require_permission
from app.identity.models import ClearanceLevel, Permission, Principal, ResourceScope, Role


router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


def _scope(record: ArtifactRecord) -> ResourceScope | None:
    if not record.organization_id or not record.workspace_id:
        return None
    return ResourceScope(
        resource_id=record.id, organization_id=record.organization_id,
        owner_id=record.owner_id, workspace_id=record.workspace_id,
        department_id=record.department_id,
        classification=ClearanceLevel.parse(record.classification),
        allowed_roles=[Role(value) for value in json.loads(record.allowed_roles_json or "[]")],
        allowed_users=list(json.loads(record.allowed_users_json or "[]")),
    )


def _allowed(principal: Principal, record: ArtifactRecord) -> bool:
    if get_settings().auth_mode.lower() != "local":
        return True
    scope = _scope(record)
    return bool(scope and AuthorizationService().can_read_artifact(principal, scope).allowed)


@router.get("")
async def list_artifacts(
    principal: Principal = Depends(require_permission(Permission.artifact_read)),
    db: Session = Depends(get_db),
) -> list[dict[str, object]]:
    records = db.query(ArtifactRecord).order_by(ArtifactRecord.created_at.desc()).limit(100).all()
    records = [row for row in records if _allowed(principal, row)]
    return [{"id": row.id, "name": row.name, "media_type": row.media_type, "size": row.size, "run_id": row.run_id} for row in records]


@router.get("/{artifact_id}")
async def download_artifact(
    artifact_id: str,
    principal: Principal = Depends(require_permission(Permission.artifact_read)),
    db: Session = Depends(get_db),
) -> FileResponse:
    record = db.get(ArtifactRecord, artifact_id)
    if not record or not _allowed(principal, record):
        AuditLogger(db, principal).log("system", "authorization_denied", "Artifact access denied.", {"resource_type": "artifact", "decision": "ARTIFACT_ACCESS_DENIED"})
        raise HTTPException(status_code=404, detail="Artifact not found")
    AuditLogger(db, principal).log(record.run_id or "system", "artifact_accessed", "Authorized artifact access.", {"artifact_id": record.id})
    service = ArtifactService(db, get_settings().workspace_root / "artifacts")
    try:
        path = service.resolve(record)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file missing")
    return FileResponse(path, media_type=record.media_type, filename=record.name)
