from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import get_settings
from app.multimodal.ocr import LocalOCRService
from app.tools.file_tools import SafeWorkspace


router = APIRouter(prefix="/api/ocr", tags=["ocr"])


class OCRRequest(BaseModel):
    path: str


@router.post("")
async def run_ocr(payload: OCRRequest) -> dict[str, object]:
    try:
        path = SafeWorkspace(get_settings().workspace_root).resolve(payload.path, must_exist=True)
        if path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            raise ValueError("OCR accepts PDF and common raster image formats")
        return LocalOCRService().extract(path).model_dump()
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

