from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agent.state import AgentRunState
from app.audit.logger import AuditLogger
from app.capsules.builder import EvidenceCapsuleBuilder
from app.capsules.models import CapsuleState
from app.capsules.verifier import EvidenceCapsuleVerifier
from app.core.config import get_settings
from app.core.database import AgentRunRecord, EvidenceCapsuleRecord, TaskEventRecord, get_db
from app.core.events import task_event_broker
from uuid import uuid4
from app.workcells.defaults import configured_workcell_registry
from app.identity.authorization import AuthorizationService
from app.identity.dependencies import require_permission
from app.identity.models import ClearanceLevel, Permission, Principal, ResourceScope, Role


task_capsule_router = APIRouter(prefix="/api/tasks", tags=["capsules"])
capsule_router = APIRouter(prefix="/api/capsules", tags=["capsules"])


class BuildCapsuleRequest(BaseModel):
    inputs: list[dict[str, object]] = Field(default_factory=list, max_length=20)


def _capsule_root(record: EvidenceCapsuleRecord) -> Path:
    root = get_settings().capsules_root.resolve()
    path = (root / record.path).resolve()
    if root not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid capsule path")
    return path


def _metadata(record: EvidenceCapsuleRecord) -> dict[str, object]:
    manifest = json.loads(record.manifest_json or "{}")
    return {
        "id": record.id, "task_id": record.task_id, "state": record.state,
        "capsule_root_hash": record.capsule_root_hash,
        "signature_status": record.signature_status,
        "manifest": manifest,
        "verify_url": f"/api/capsules/{record.id}/verify",
        "download_url": f"/api/capsules/{record.id}/download",
    }


def _scope(record: EvidenceCapsuleRecord | AgentRunRecord) -> ResourceScope | None:
    if not record.organization_id or not record.workspace_id:
        return None
    return ResourceScope(
        resource_id=record.id, organization_id=record.organization_id,
        owner_id=record.owner_id, workspace_id=record.workspace_id,
        department_id=record.department_id,
        classification=ClearanceLevel.parse(record.classification),
        allowed_roles=[Role(value) for value in json.loads(getattr(record, "allowed_roles_json", "[]") or "[]")],
        allowed_users=list(json.loads(getattr(record, "allowed_users_json", "[]") or "[]")),
    )


def _assert_access(principal: Principal, record: EvidenceCapsuleRecord, permission: Permission) -> None:
    if get_settings().auth_mode.lower() != "local":
        return
    scope = _scope(record)
    decision = AuthorizationService().authorize(principal, permission, scope) if scope else None
    if not decision or not decision.allowed:
        raise HTTPException(status_code=404, detail="Evidence Capsule not found")


async def _publish_capsule_event(db: Session, task_id: str, event_type: str, payload: dict[str, object]) -> None:
    event = await task_event_broker.publish(task_id, event_type, payload)
    db.add(TaskEventRecord(
        id=str(uuid4()), task_id=task_id, event_type=event_type,
        payload_json=json.dumps(payload, ensure_ascii=False, default=str),
        created_at=datetime.fromisoformat(event["timestamp"]),
    ))
    db.commit()


@task_capsule_router.post("/{task_id}/capsule")
async def build_task_capsule(
    task_id: str,
    payload: BuildCapsuleRequest | None = None,
    principal: Principal = Depends(require_permission(Permission.capsule_create)),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    row = db.get(AgentRunRecord, task_id)
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    if get_settings().auth_mode.lower() == "local":
        task_scope = _scope(row)
        task_access = AuthorizationService().authorize(principal, Permission.task_read, task_scope) if task_scope else None
        if not task_access or not task_access.allowed:
            raise HTTPException(status_code=404, detail="Task not found")
    task = AgentRunState.model_validate_json(row.state_json)
    if task.status.value not in {"completed", "waiting_for_approval"}:
        raise HTTPException(status_code=409, detail="Capsules require a completed Workcell task")
    if not task.workcell_id:
        raise HTTPException(status_code=400, detail="CAPSULE_BUILD_FAILED: task was not executed by a Workcell")
    registry = configured_workcell_registry(get_settings())
    try:
        definition = registry.get(task.workcell_id, task.workcell_version)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    settings = get_settings()
    await _publish_capsule_event(db, task_id, "capsule_build_started", {"task_id": task_id})
    try:
        record = EvidenceCapsuleBuilder(
            db, settings.capsules_root, settings.workspace_root / "artifacts"
        ).build(task, definition, inputs=(payload.inputs if payload else []))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"CAPSULE_BUILD_FAILED: {exc}") from exc
    await _publish_capsule_event(db, task_id, "capsule_created", {
        "capsule_id": record.id, "capsule_root_hash": record.capsule_root_hash,
    })
    return _metadata(record)


@task_capsule_router.get("/{task_id}/capsule")
async def get_task_capsule(
    task_id: str,
    principal: Principal = Depends(require_permission(Permission.capsule_read)),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    record = db.query(EvidenceCapsuleRecord).filter(
        EvidenceCapsuleRecord.task_id == task_id
    ).order_by(EvidenceCapsuleRecord.created_at.desc()).first()
    if not record:
        raise HTTPException(status_code=404, detail="Evidence Capsule not found")
    _assert_access(principal, record, Permission.capsule_read)
    return _metadata(record)


@capsule_router.get("")
async def list_capsules(
    principal: Principal = Depends(require_permission(Permission.capsule_read)),
    db: Session = Depends(get_db),
) -> list[dict[str, object]]:
    records = db.query(EvidenceCapsuleRecord).order_by(EvidenceCapsuleRecord.created_at.desc()).limit(100).all()
    if get_settings().auth_mode.lower() == "local":
        records = [row for row in records if (_scope(row) and AuthorizationService().can_read_capsule(principal, _scope(row)).allowed)]
    return [_metadata(row) for row in records]


@capsule_router.get("/{capsule_id}")
async def get_capsule(
    capsule_id: str,
    principal: Principal = Depends(require_permission(Permission.capsule_read)),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    record = db.get(EvidenceCapsuleRecord, capsule_id)
    if not record:
        raise HTTPException(status_code=404, detail="Evidence Capsule not found")
    _assert_access(principal, record, Permission.capsule_read)
    return _metadata(record)


@capsule_router.post("/{capsule_id}/verify")
async def verify_capsule(
    capsule_id: str,
    principal: Principal = Depends(require_permission(Permission.capsule_verify)),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    record = db.get(EvidenceCapsuleRecord, capsule_id)
    if not record:
        raise HTTPException(status_code=404, detail="Evidence Capsule not found")
    _assert_access(principal, record, Permission.capsule_verify)
    settings = get_settings()
    result = EvidenceCapsuleVerifier(unsigned_allowed=settings.unsigned_capsules_allowed).verify(_capsule_root(record))
    record.state = CapsuleState.verified.value if result.status.value == "VALID" else CapsuleState.invalid.value
    record.signature_status = result.signature_status.value
    record.updated_at = datetime.now(timezone.utc)
    db.commit()
    AuditLogger(db, principal).log(record.task_id, "capsule_verified" if result.status.value == "VALID" else "capsule_verification_failure", f"Capsule {record.id}: {result.status.value}", result.model_dump(mode="json"))
    await _publish_capsule_event(db, record.task_id, "capsule_verified" if result.status.value == "VALID" else "capsule_invalid", {
        "capsule_id": record.id, "status": result.status.value,
        "failures": [item.model_dump(mode="json") for item in result.failures],
    })
    return result.model_dump(mode="json")


@capsule_router.get("/{capsule_id}/download")
async def download_capsule(
    capsule_id: str,
    principal: Principal = Depends(require_permission(Permission.capsule_read)),
    db: Session = Depends(get_db),
) -> Response:
    record = db.get(EvidenceCapsuleRecord, capsule_id)
    if not record:
        raise HTTPException(status_code=404, detail="Evidence Capsule not found")
    _assert_access(principal, record, Permission.capsule_read)
    root = _capsule_root(record)
    if not root.is_dir() or record.state == CapsuleState.building.value:
        raise HTTPException(status_code=409, detail="Evidence Capsule is not available for download")
    output = io.BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
            archive.write(path, arcname=f"{record.id}/{path.relative_to(root).as_posix()}")
    return Response(
        content=output.getvalue(), media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="evidence-capsule-{record.id}.zip"'},
    )
