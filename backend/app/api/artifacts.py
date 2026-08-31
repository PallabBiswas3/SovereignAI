from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.artifacts.service import ArtifactService
from app.core.config import get_settings
from app.core.database import ArtifactRecord, get_db


router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


@router.get("")
async def list_artifacts(db: Session = Depends(get_db)) -> list[dict[str, object]]:
    records = db.query(ArtifactRecord).order_by(ArtifactRecord.created_at.desc()).limit(100).all()
    return [{"id": row.id, "name": row.name, "media_type": row.media_type, "size": row.size, "run_id": row.run_id} for row in records]


@router.get("/{artifact_id}")
async def download_artifact(artifact_id: str, db: Session = Depends(get_db)) -> FileResponse:
    record = db.get(ArtifactRecord, artifact_id)
    if not record:
        raise HTTPException(status_code=404, detail="Artifact not found")
    service = ArtifactService(db, get_settings().workspace_root / "artifacts")
    try:
        path = service.resolve(record)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file missing")
    return FileResponse(path, media_type=record.media_type, filename=record.name)

