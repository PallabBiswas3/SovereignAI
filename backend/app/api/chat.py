from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import Conversation, Message, get_db
from app.llm.ollama_provider import OllamaProvider
from app.router.model_registry import ModelRegistry
from app.router.model_router import ModelRouter
from app.router.schemas import RoutingDecision


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


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    conversation_id = payload.conversation_id or str(uuid4())
    if payload.conversation_id is None:
        db.add(Conversation(id=conversation_id, title=payload.message[:80]))
    db.add(Message(id=str(uuid4()), conversation_id=conversation_id, role="user", content=payload.message))

    settings = get_settings()
    registry = ModelRegistry(settings.models_config)
    routing = ModelRouter(registry).route(payload.message, payload.model_override)
    selected = registry.get(routing.model_id)
    provider = OllamaProvider(selected.endpoint, settings.allow_deterministic_fallback)
    result = await provider.generate(
        payload.message,
        selected.model_tag,
        "You are SovereignAI, a local enterprise assistant. Be concise and never invent sources.",
    )
    db.add(Message(id=str(uuid4()), conversation_id=conversation_id, role="assistant", content=result.text))
    db.commit()
    return ChatResponse(
        conversation_id=conversation_id,
        response=result.text,
        model=result.model,
        provider=result.provider,
        fallback=result.fallback,
        routing=routing,
    )
