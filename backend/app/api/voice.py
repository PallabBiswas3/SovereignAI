from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from starlette.requests import ClientDisconnect
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.identity.dependencies import get_current_principal, require_permission
from app.identity.models import Permission, Principal
from app.multimodal.speech import FORMATS, SpeechError, get_speech_service

router = APIRouter(prefix="/api/voice", tags=["voice"])


@router.get("/config")
async def voice_config(
    response: Response,
    principal: Principal = Depends(require_permission(Permission.task_create)),
):
    settings = get_settings()
    response.headers["Cache-Control"] = "no-store"
    return {
        "max_duration_seconds": settings.voice_max_duration_seconds,
        "max_upload_bytes": settings.voice_max_upload_bytes,
        "timeout_seconds": settings.voice_timeout_seconds,
        "language": settings.voice_language,
        "mime_types": list(FORMATS),
    }


@router.post("/transcribe")
async def transcribe(
    request: Request, response: Response,
    principal: Principal = Depends(require_permission(Permission.task_create)),
    db: Session = Depends(get_db),
):
    """Bound the raw audio stream before allocating; no multipart spool files."""
    settings = get_settings()
    mime = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if mime not in FORMATS:
        raise HTTPException(415, detail={"code": "UNSUPPORTED_AUDIO_FORMAT"})
    length = request.headers.get("content-length")
    if length is not None:
        try:
            size = int(length)
            if size < 0:
                raise ValueError
        except ValueError:
            raise HTTPException(400, detail={"code": "INVALID_CONTENT_LENGTH"}) from None
        if size > settings.voice_max_upload_bytes:
            raise HTTPException(413, detail={"code": "AUDIO_TOO_LARGE"})
    data = bytearray()
    try:
        async with asyncio.timeout(settings.voice_upload_timeout_seconds):
            async for chunk in request.stream():
                if len(data) + len(chunk) > settings.voice_max_upload_bytes:
                    raise HTTPException(413, detail={"code": "AUDIO_TOO_LARGE"})
                data.extend(chunk)
        # Recheck expiration/revocation after a slow upload and again after inference.
        def reauthorize():
            db.expire_all()
            current = get_current_principal(request, db)
            if current.user_id != principal.user_id or not current.has_permission(Permission.task_create):
                raise HTTPException(403, detail={"code": "ACCESS_DENIED"})

        reauthorize()
        result = await get_speech_service().transcribe(bytes(data), mime)
        reauthorize()
    except TimeoutError:
        raise HTTPException(408, detail={"code": "AUDIO_UPLOAD_TIMEOUT"}) from None
    except ClientDisconnect:
        raise HTTPException(400, detail={"code": "AUDIO_UPLOAD_INTERRUPTED"}) from None
    except SpeechError as exc:
        raise HTTPException(exc.status, detail={"code": exc.code}) from None
    finally:
        data.clear()
    response.headers["Cache-Control"] = "no-store"
    return result
