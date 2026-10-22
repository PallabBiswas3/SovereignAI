from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import Conversation, Message, get_db
from app.llm.factory import configured_local_provider
from app.router.model_registry import ModelRegistry
from app.router.model_router import ModelRouter
from app.router.schemas import RoutingDecision
from app.identity.dependencies import require_permission
from app.identity.models import Permission, Principal
from app.governance.injection import PromptInjectionScanner
from app.governance.pii import PIIDetector
from app.governance.policy_engine import PolicyEngine, GovernanceDecision
from app.integrations.clients import ControlPlaneClient, IntegrationServiceError
from app.integrations.orchestrator import IndustrialIntegrationOrchestrator, RELEASE_ACTIONS


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
    released: bool = False
    governance_action: str = "block"


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    principal: Principal = Depends(require_permission(Permission.task_create)),
    db: Session = Depends(get_db),
) -> ChatResponse:
    settings = get_settings()
    conversation_id = payload.conversation_id or str(uuid4())
    workspace = principal.workspace_ids[0] if principal.workspace_ids else None
    if settings.auth_mode == "local" and (not workspace or workspace == "*"
            or not principal.organization_id or principal.organization_id == "*"):
        raise HTTPException(status_code=403, detail="Explicit chat scope required")
    if payload.conversation_id:
        conversation = db.get(Conversation, conversation_id)
        if conversation is None or (settings.auth_mode == "local" and (
                conversation.organization_id != principal.organization_id
                or conversation.owner_id != principal.user_id
                or not conversation.workspace_id
                or conversation.workspace_id not in principal.workspace_ids)):
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conversation = Conversation(id=conversation_id, title=payload.message[:80],
                                    organization_id=principal.organization_id,
                                    owner_id=principal.user_id, workspace_id=workspace)

    engine = PolicyEngine(settings.policies_config)
    decision = engine.decide(engine.get("internal_assistant"),
                             pii_count=len(PIIDetector().detect(payload.message)),
                             injection_count=len(PromptInjectionScanner().scan(payload.message)))
    if decision == GovernanceDecision.block:
        raise HTTPException(status_code=403, detail="Request blocked by governance policy")
    registry = ModelRegistry(settings.models_config)
    routing = ModelRouter(registry).route(payload.message, payload.model_override)
    selected = registry.get(routing.model_id)
    runtime = configured_local_provider(settings, model_definition=selected)
    controlplane = ControlPlaneClient(settings.controlplane_url,
        timeout=settings.integration_timeout_seconds, max_retries=settings.integration_max_retries)
    common = {"profile": "internal_assistant", "prompt": payload.message,
              "consequential": True, "industry": "industrial",
              "metadata": {"principal_id": principal.user_id, "organization_id": principal.organization_id}}
    try:
        precheck = await controlplane.precheck({**common, "id": f"{uuid4()}-precheck"})
        if IndustrialIntegrationOrchestrator._action(precheck) not in RELEASE_ACTIONS:
            raise HTTPException(status_code=403, detail="Request held by ControlPlane")
        result = await runtime.provider.generate(
            payload.message, runtime.model,
            "You are SovereignAI, a local enterprise assistant. Be concise and never invent sources.",
        )
        report = await controlplane.check({**common, "id": f"{uuid4()}-release",
                                          "response": result.text, "context": "",
                                          "grounding_evidence": []})
    except IntegrationServiceError as exc:
        raise HTTPException(status_code=503, detail="ControlPlane unavailable; response held") from exc
    action = IndustrialIntegrationOrchestrator._action(report)
    released = action in RELEASE_ACTIONS and bool(report.get("final_response"))
    response = str(report["final_response"]) if released else "Response held by governance policy."
    db.add(conversation)
    db.add(Message(id=str(uuid4()), conversation_id=conversation_id, role="user", content=payload.message))
    db.add(Message(id=str(uuid4()), conversation_id=conversation_id, role="assistant", content=response))
    db.commit()
    return ChatResponse(
        conversation_id=conversation_id,
        response=response,
        model=result.model,
        provider=result.provider,
        fallback=result.fallback,
        routing=routing,
        released=released,
        governance_action=action,
    )
