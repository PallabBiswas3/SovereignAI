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
    allow_credentials=False,
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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "sovereign-backend"}
