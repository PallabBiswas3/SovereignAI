from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import get_settings
from app.workcells.defaults import configured_workcell_registry
from app.identity.dependencies import require_permission
from app.identity.models import Permission, Principal


router = APIRouter(prefix="/api/workcells", tags=["workcells"])


@router.get("")
async def list_workcells(
    principal: Principal = Depends(require_permission(Permission.task_read)),
) -> list[dict[str, object]]:
    return [item.model_dump(mode="json") for item in configured_workcell_registry(get_settings()).list()]


@router.get("/{workcell_id}")
async def get_workcell(
    workcell_id: str,
    principal: Principal = Depends(require_permission(Permission.task_read)),
) -> dict[str, object]:
    registry = configured_workcell_registry(get_settings())
    try:
        definition = registry.get(workcell_id, require_ready=False)
        validation = registry.validation(workcell_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "manifest": definition.manifest.model_dump(mode="json"),
        "workflow": definition.workflow.model_dump(mode="json"),
        "content_hash": definition.content_hash,
        "prompt_manifest": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(definition.prompt_hashes.items())
        ],
        "validation": validation.model_dump(mode="json"),
    }


@router.post("/{workcell_id}/validate")
async def validate_workcell(
    workcell_id: str,
    principal: Principal = Depends(require_permission(Permission.workcell_manage)),
) -> dict[str, object]:
    registry = configured_workcell_registry(get_settings())
    try:
        return registry.validation(workcell_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
