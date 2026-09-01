from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.multimodal.vision import OllamaVisionProvider
from app.router.model_registry import ModelRegistry
from app.tools.file_tools import SafeWorkspace
from app.resources.cache import get_cache_backend


router = APIRouter(prefix="/api/vision", tags=["vision"])


class VisionRequest(BaseModel):
    path: str
    prompt: str = Field(default="Describe relevant components and observations.", max_length=5_000)


@router.post("/analyze")
async def analyze_image(payload: VisionRequest) -> dict[str, object]:
    settings = get_settings()
    try:
        path = SafeWorkspace(settings.workspace_root).resolve(payload.path, must_exist=True)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            raise ValueError("Vision analysis requires a common raster image")
        model = ModelRegistry(settings.models_config).get("vision")
        cache = get_cache_backend() if settings.cache_enabled else None
        result = await OllamaVisionProvider(
            model.endpoint, model.model_tag, cache=cache
        ).analyze_image(path, payload.prompt)
        return result.model_dump()
    except (ValueError, FileNotFoundError, OSError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
