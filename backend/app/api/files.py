from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.tools.file_tools import ALLOWED_EXTENSIONS, principal_workspace
from app.identity.dependencies import require_permission
from app.identity.models import Permission, Principal


router = APIRouter(prefix="/api/files", tags=["files"])


def _safe_name(name: str | None) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(name or "upload").name).strip()
    if not cleaned or cleaned in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return cleaned


def _workspace(principal: Principal):
    try:
        return principal_workspace(get_settings().workspace_root, principal)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail={"code": "ACCESS_SCOPE_REQUIRED"}) from exc


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
    workspace = _workspace(principal)
    relative = f"uploads/{uuid4().hex}/{name}"
    target = workspace.resolve(relative)
    target.parent.mkdir(parents=True, exist_ok=False)
    with target.open("xb") as output:
        output.write(content)
    return {"name": name, "size": len(content), "path": relative}


@router.get("")
async def list_files(
    principal: Principal = Depends(require_permission(Permission.task_read)),
) -> list[dict[str, object]]:
    workspace = _workspace(principal)
    root = workspace.resolve("uploads")
    root.mkdir(parents=True, exist_ok=True)
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        relative = path.relative_to(workspace.root).as_posix()
        try:
            workspace.resolve(relative, must_exist=True)
        except (ValueError, FileNotFoundError):
            continue
        files.append({"name": path.name, "size": path.stat().st_size, "path": relative})
    return files
