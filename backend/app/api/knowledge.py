from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.rag.embeddings import configured_embedding_provider
from app.rag.ingestion import KnowledgeIngestionService
from app.rag.retrieval import LocalRetriever
from app.tools.file_tools import SafeWorkspace
from app.governance.injection import PromptInjectionScanner


router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


class IngestRequest(BaseModel):
    path: str
    department: str | None = None
    classification: str = "internal"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5_000)
    limit: int = Field(default=5, ge=1, le=20)


@router.post("/ingest")
async def ingest(payload: IngestRequest, db: Session = Depends(get_db)) -> dict[str, object]:
    settings = get_settings()
    workspace = SafeWorkspace(settings.knowledge_root)
    try:
        path = workspace.resolve(payload.path, must_exist=True)
        scan_text = path.read_text(encoding="utf-8", errors="ignore") if path.suffix.lower() in {".txt", ".md", ".csv", ".json"} else ""
        injection_findings = PromptInjectionScanner().scan(scan_text)
        embeddings = configured_embedding_provider()
        document = KnowledgeIngestionService(db, embeddings).ingest(
            path, {"department": payload.department, "classification": payload.classification}
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": document.id, "filename": document.filename, "chunks": document.chunk_count,
            "embedding_provider": document.embedding_provider,
            "injection_findings": injection_findings, "content_treated_as_data": True}


@router.post("/answer")
async def grounded_answer(payload: SearchRequest, db: Session = Depends(get_db)) -> dict[str, object]:
    results = LocalRetriever(db, configured_embedding_provider()).search(payload.query, payload.limit)
    if not results or results[0].score < 0.18:
        return {"answer": "I could not establish this from the available internal documents.", "grounded": False, "grounding_score": round(results[0].score, 4) if results else 0.0, "sources": []}
    top = results[0]
    return {"answer": top.text, "grounded": True, "grounding_score": round(top.score, 4), "sources": [top.source]}


@router.post("/search")
async def search(payload: SearchRequest, db: Session = Depends(get_db)) -> dict[str, object]:
    results = LocalRetriever(db, configured_embedding_provider()).search(payload.query, payload.limit)
    return {"query": payload.query, "results": [{"text": item.text, "score": round(item.score, 4), "source": item.source} for item in results]}
