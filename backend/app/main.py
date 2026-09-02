from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.models import router as models_router
from app.api.tasks import router as tasks_router
from app.api.files import router as files_router
from app.api.knowledge import router as knowledge_router
from app.api.artifacts import router as artifacts_router
from app.api.demo import router as demo_router
from app.api.ocr import router as ocr_router
from app.api.vision import router as vision_router
from app.api.governance import router as governance_router
from app.api.monitoring import router as monitoring_router
from app.api.evaluation import router as evaluation_router
from app.api.workcells import router as workcells_router
from app.api.capsules import capsule_router, task_capsule_router
from app.api.auth import router as auth_router
from app.api.organization import router as organization_router
from app.identity.provider import LocalIdentityProvider
import hmac
from fastapi import Request
from fastapi.responses import JSONResponse
from app.core.database import SessionLocal
from app.core.config import get_settings
from app.core.database import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router)
app.include_router(models_router)
app.include_router(tasks_router)
app.include_router(files_router)
app.include_router(knowledge_router)
app.include_router(artifacts_router)
app.include_router(demo_router)
app.include_router(ocr_router)
app.include_router(vision_router)
app.include_router(governance_router)
app.include_router(monitoring_router)
app.include_router(evaluation_router)
app.include_router(workcells_router)
app.include_router(task_capsule_router)
app.include_router(capsule_router)
app.include_router(auth_router)
app.include_router(organization_router)


@app.middleware("http")
async def local_csrf_protection(request: Request, call_next):
    settings = get_settings()
    if settings.auth_mode.lower() == "local" and request.method.upper() not in {"GET", "HEAD", "OPTIONS"} and request.url.path != "/api/auth/login":
        raw_token = request.cookies.get(settings.auth_cookie_name)
        if raw_token:
            cookie_csrf = request.cookies.get(settings.auth_csrf_cookie_name, "")
            header_csrf = request.headers.get("X-CSRF-Token", "")
            with SessionLocal() as session:
                record = LocalIdentityProvider(session, settings.access_config).session_record(raw_token)
            expected = record.csrf_token if record else ""
            if not expected or not hmac.compare_digest(cookie_csrf, expected) or not hmac.compare_digest(header_csrf, expected):
                return JSONResponse(status_code=403, content={"detail": {"code": "CSRF_VALIDATION_FAILED"}})
    return await call_next(request)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "sovereign-backend"}
