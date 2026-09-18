from __future__ import annotations

from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import Conversation, Message, get_db
from app.llm.provider_factory import create_local_model_provider
from app.router.model_registry import ModelRegistry
from app.router.model_router import ModelRouter
from app.router.schemas import RoutingDecision
from app.identity.dependencies import require_permission
from app.identity.models import Permission, Principal


router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=50_000)
    conversation_id: str | None = None
    model_override: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    model: str
    provider: str
    fallback: bool
    routing: RoutingDecision
    runtime_metrics: dict[str, Any] = Field(default_factory=dict)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    principal: Principal = Depends(require_permission(Permission.task_create)),
    db: Session = Depends(get_db),
) -> ChatResponse:
    request_started = perf_counter()

    setup_started = perf_counter()
    conversation_id = payload.conversation_id or str(uuid4())
    if payload.conversation_id is None:
        db.add(Conversation(id=conversation_id, title=payload.message[:80]))
    db.add(Message(id=str(uuid4()), conversation_id=conversation_id, role="user", content=payload.message))
    request_setup_seconds = perf_counter() - setup_started

    routing_started = perf_counter()
    settings = get_settings()
    registry = ModelRegistry(settings.models_config)
    routing = ModelRouter(registry).route(payload.message, payload.model_override)
    selected = registry.get(routing.model_id)
    routing_seconds = perf_counter() - routing_started

    provider_started = perf_counter()
    provider = create_local_model_provider(
        settings,
        model=selected.model_tag,
        ollama_endpoint=selected.endpoint,
        role=selected.role,
        memory_requirement=selected.memory_requirement,
        execution_mode="FAST",
    )
    provider_setup_seconds = perf_counter() - provider_started

    generation_started = perf_counter()
    result = await provider.generate(
        payload.message,
        selected.model_tag,
        "You are SovereignAI, a local enterprise assistant. Be concise and never invent sources.",
    )
    generation_seconds = perf_counter() - generation_started

    persistence_started = perf_counter()
    db.add(Message(id=str(uuid4()), conversation_id=conversation_id, role="assistant", content=result.text))
    db.commit()
    persistence_commit_seconds = perf_counter() - persistence_started
    endpoint_total_seconds = perf_counter() - request_started

    runtime_metrics: dict[str, Any] = {
        "endpoint_total_seconds": round(endpoint_total_seconds, 6),
        "request_setup_seconds": round(request_setup_seconds, 6),
        "routing_seconds": round(routing_seconds, 6),
        "provider_setup_seconds": round(provider_setup_seconds, 6),
        "generation_seconds": round(generation_seconds, 6),
        "persistence_commit_seconds": round(persistence_commit_seconds, 6),
        "measured_phase_seconds": round(
            request_setup_seconds
            + routing_seconds
            + provider_setup_seconds
            + generation_seconds
            + persistence_commit_seconds,
            6,
        ),
        "provider_generation": result.runtime_stats,
    }

    provider_total = result.runtime_stats.get("total_duration_seconds") if isinstance(result.runtime_stats, dict) else None
    if isinstance(provider_total, (int, float)):
        runtime_metrics["application_overhead_excluding_model_seconds"] = round(
            max(0.0, endpoint_total_seconds - float(provider_total)), 6
        )

    return ChatResponse(
        conversation_id=conversation_id,
        response=result.text,
        model=result.model,
        provider=result.provider,
        fallback=result.fallback,
        routing=routing,
        runtime_metrics=runtime_metrics,
    )
