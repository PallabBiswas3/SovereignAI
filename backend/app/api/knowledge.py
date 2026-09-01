from __future__ import annotations

from pathlib import Path
import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.rag.embeddings import configured_embedding_provider
from app.rag.ingestion import KnowledgeIngestionService
from app.rag.factory import configured_hybrid_retriever
from app.tools.file_tools import SafeWorkspace
from app.governance.injection import PromptInjectionScanner
from app.resources.cache import get_cache_backend


router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


class IngestRequest(BaseModel):
    path: str
    department: str | None = None
    classification: str = "internal"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5_000)
    limit: int = Field(default=5, ge=1, le=20)
    execution_mode: str = "STANDARD"


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
    settings = get_settings()
    cache = get_cache_backend() if settings.cache_enabled else None
    retriever = configured_hybrid_retriever(
        db, cache=cache, settings=settings, execution_mode=payload.execution_mode
    )
    results = retriever.search(payload.query, payload.limit)
    dense_score = float(results[0].scores.get("dense") or 0.0) if results else 0.0
    query_terms = {value for value in re.findall(r"[a-z]{4,}", payload.query.lower())
                   if value not in {"what", "which", "from", "this", "that", "correct"}}
    evidence_terms = set(re.findall(r"[a-z]{4,}", results[0].text.lower())) if results else set()
    lexical_support = len(query_terms & evidence_terms) / max(1, len(query_terms))
    if not results or (dense_score < 0.45 and lexical_support < 0.34):
        return {"answer": "I could not establish this from the available authorized evidence.", "grounded": False, "support_status": "INSUFFICIENT_EVIDENCE", "grounding_score": round(max(dense_score, lexical_support), 4), "sources": []}
    top = results[0]
    return {"answer": top.text, "grounded": True, "support_status": "SUPPORTED", "grounding_score": round(max(dense_score, lexical_support), 4), "sources": [top.source], "retrieval": top.to_dict()}


@router.post("/search")
async def search(payload: SearchRequest, db: Session = Depends(get_db)) -> dict[str, object]:
    settings = get_settings()
    cache = get_cache_backend() if settings.cache_enabled else None
    retriever = configured_hybrid_retriever(
        db, cache=cache, settings=settings, execution_mode=payload.execution_mode
    )
    from app.rag.decomposition import ModeAwareRetrievalPipeline
    pipeline = ModeAwareRetrievalPipeline(retriever, settings.max_retrieval_subqueries)
    results = pipeline.search(payload.query, payload.execution_mode, payload.limit)
    return {"query": payload.query, "subqueries": pipeline.last_subqueries,
            "results": [item.to_dict() for item in results],
            "telemetry": retriever.last_telemetry.to_dict()}
