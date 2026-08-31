from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.artifacts.service import ArtifactService
from app.core.config import get_settings
from app.core.database import get_db
from app.rag.embeddings import configured_embedding_provider
from app.rag.ingestion import KnowledgeIngestionService
from app.rag.retrieval import LocalRetriever
from app.tools.file_tools import SafeWorkspace
from app.workflows.inspection import InspectionWorkflow
from app.workflows.coding import CodingWorkflow
from app.sandbox.executor import DockerSandboxExecutor
from app.llm.ollama_provider import OllamaProvider
from app.router.model_registry import ModelRegistry


router = APIRouter(prefix="/api/demo", tags=["demo"])


class InspectionDemoRequest(BaseModel):
    inspection_path: str = "uploads/Pump_Inspection_Report.pdf"
    sop_path: str = "Maintenance_SOP.pdf"


class CodingDemoRequest(BaseModel):
    csv_path: str = "uploads/pump_sensor_readings.csv"


@router.post("/inspection")
async def run_inspection_demo(payload: InspectionDemoRequest, db: Session = Depends(get_db)) -> dict[str, object]:
    settings = get_settings()
    try:
        inspection = SafeWorkspace(settings.workspace_root).resolve(payload.inspection_path, must_exist=True)
        sop = SafeWorkspace(settings.knowledge_root).resolve(payload.sop_path, must_exist=True)
        embeddings = configured_embedding_provider()
        KnowledgeIngestionService(db, embeddings).ingest(sop, {"department": "maintenance", "classification": "internal"})
        output = settings.workspace_root / "artifacts" / f"Approval_Note_{uuid4().hex[:8]}.docx"
        analysis = InspectionWorkflow(LocalRetriever(db, embeddings)).analyze(inspection, output)
        artifact = ArtifactService(db, settings.workspace_root / "artifacts").register(output)
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {**analysis.model_dump(), "artifact": {"id": artifact.id, "name": artifact.name, "size": artifact.size}}


@router.post("/coding")
async def run_coding_demo(payload: CodingDemoRequest, db: Session = Depends(get_db)) -> dict[str, object]:
    settings = get_settings()
    try:
        csv_path = SafeWorkspace(settings.workspace_root).resolve(payload.csv_path, must_exist=True)
        if csv_path.suffix.lower() != ".csv":
            raise ValueError("Coding demo requires a CSV input")
        artifact_root = settings.workspace_root / "artifacts"
        coder = ModelRegistry(settings.models_config).get("coder")
        result = await CodingWorkflow(
            DockerSandboxExecutor(settings.workspace_root / "sandbox"),
            OllamaProvider(coder.endpoint, settings.allow_deterministic_fallback),
            coder.model_tag,
        ).run(csv_path, artifact_root, "Analyze anomalies and create a reusable verified Python program")
        service = ArtifactService(db, artifact_root)
        paths = [result.source_path, result.report_path, *result.result_paths]
        artifacts = [service.register(Path(path)) for path in paths if path]
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {**result.model_dump(), "artifacts": [{"id": item.id, "name": item.name, "size": item.size} for item in artifacts]}
