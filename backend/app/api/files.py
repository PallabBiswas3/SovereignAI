from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.tools.file_tools import ALLOWED_EXTENSIONS, SafeWorkspace
from app.identity.dependencies import require_permission
from app.identity.models import Permission, Principal


router = APIRouter(prefix="/api/files", tags=["files"])


def _safe_name(name: str | None) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(name or "upload").name).strip()
    if not cleaned or cleaned in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return cleaned


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    principal: Principal = Depends(require_permission(Permission.task_create)),
) -> dict[str, object]:
    settings = get_settings()
    name = _safe_name(file.filename)
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Upload exceeds configured limit")
    workspace = SafeWorkspace(settings.workspace_root / "uploads")
    target = workspace.resolve(name)
    target.write_bytes(content)
    return {"name": name, "size": len(content), "path": f"uploads/{name}"}


@router.get("")
async def list_files(
    principal: Principal = Depends(require_permission(Permission.task_read)),
) -> list[dict[str, object]]:
    root = get_settings().workspace_root / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return [{"name": path.name, "size": path.stat().st_size, "path": f"uploads/{path.name}"} for path in root.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS]
