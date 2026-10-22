from __future__ import annotations

import json
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session
from sqlalchemy import update

from app.audit.logger import AuditLogger
from app.core.config import get_settings
from app.core.database import AgentRunRecord, AuditEventRecord, HumanApprovalRecord, MaintenanceDraftRecord, get_db
from app.governance.action_guard import ActionGuard
from app.governance.injection import PromptInjectionScanner
from app.governance.pii import PIIDetector
from app.governance.policy_engine import PolicyEngine
from app.tools.registry import create_default_registry, create_agent_registry
from app.identity import ContentIdentityService
from app.identity.authorization import AuthorizationService
from app.identity.dependencies import require_permission
from app.identity.models import Permission, Principal
from app.identity.provider import LocalIdentityProvider


router = APIRouter(prefix="/api", tags=["governance"])


class GovernanceCheckRequest(BaseModel):
    text: str
    use_case: str = "internal_assistant"


class ToolProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: str
    args: dict[str, object] = Field(default_factory=dict)
    run_id: str | None = None


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approve: bool
    decided_by: str | None = None


@router.post("/governance/check")
async def governance_check(
    payload: GovernanceCheckRequest,
    principal: Principal = Depends(require_permission(Permission.task_create)),
) -> dict[str, object]:
    settings = get_settings()
    pii = PIIDetector().detect(payload.text)
    injections = PromptInjectionScanner().scan(payload.text)
    policy_engine = PolicyEngine(settings.policies_config)
    policy = policy_engine.get(payload.use_case)
    decision = policy_engine.decide(policy, pii_count=len(pii), injection_count=len(injections))
    return {"decision": decision.value, "policy": policy.name, "pii": [asdict(item) for item in pii], "injection_findings": injections}


@router.post("/tools/propose")
async def propose_tool(
    payload: ToolProposal,
    principal: Principal = Depends(require_permission(Permission.tool_execute)),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    settings = get_settings()
    action = ActionGuard(settings.tools_config).evaluate(payload.tool)
    response: dict[str, object] = action.model_dump(mode="json")
    if action.decision.value == "REQUIRE_HUMAN_APPROVAL":
        if not principal.has_permission(Permission.approval_request):
            raise HTTPException(status_code=403, detail={"code": "ACCESS_DENIED"})
        try:
            errors = create_default_registry(settings.workspace_root).validate_arguments(payload.tool, payload.args)
        except KeyError:
            errors = []  # A configured high-risk tool may be proposed but remains non-executable.
        if errors:
            raise HTTPException(status_code=422, detail={"argument_errors": errors})
        approval = HumanApprovalRecord(
            id=str(uuid4()), run_id=payload.run_id, tool=payload.tool,
            args_json=json.dumps(payload.args), risk=action.risk, status="pending",
            requester_id=principal.user_id, organization_id=principal.organization_id,
            workspace_id=next(iter(principal.workspace_ids), None),
            action_hash=ContentIdentityService().hash_json({"tool": payload.tool, "arguments": payload.args}),
        )
        db.add(approval)
        db.commit()
        response["approval_id"] = approval.id
    if payload.run_id:
        AuditLogger(db, principal).log(payload.run_id, "tool_policy", action.reason, response)
    return response


@router.post("/approvals/{approval_id}")
async def decide_approval(
    approval_id: str, payload: ApprovalDecision,
    principal: Principal = Depends(require_permission(Permission.approval_approve)),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    approval = db.get(HumanApprovalRecord, approval_id)
    if not approval:
        raise HTTPException(status_code=404, detail="Approval not found")
    if approval.status != "pending":
        raise HTTPException(status_code=409, detail="Approval already decided")
    from app.agent.state import AgentRunState, RunStatus
    from app.agent.resume import task_scope, revalidate_principal
    from app.agent.receipts import ActionContext, action_context, persist_receipt
    task_row = db.get(AgentRunRecord, approval.run_id) if approval.run_id else None
    saved_state = AgentRunState.model_validate_json(task_row.state_json) if task_row else None
    if saved_state and not AuthorizationService().authorize_resource(principal, task_scope(saved_state)).allowed:
        raise HTTPException(status_code=403, detail={"code": "APPROVAL_SCOPE_DENIED"})
    separation = AuthorizationService().can_approve_action(principal, approval.requester_id)
    if get_settings().auth_mode.lower() == "local" and not separation.allowed:
        AuditLogger(db, principal).log(approval.run_id or f"approval:{approval.id}", "authorization_denied", "Approval authority denied.", {"code": separation.reason_code})
        raise HTTPException(status_code=403, detail={"code": separation.reason_code})
    # Claim the decision durably before awaiting a tool. A concurrent approver
    # must never execute the same action twice.
    changed = db.execute(update(HumanApprovalRecord).where(HumanApprovalRecord.id == approval_id,
        HumanApprovalRecord.status == "pending", HumanApprovalRecord.args_json == approval.args_json,
        HumanApprovalRecord.action_hash == approval.action_hash).values(
            status="approved" if payload.approve else "rejected",
            execution_status="executing" if payload.approve else "not_executed",
            decided_by=principal.user_id if get_settings().auth_mode.lower() == "local" else (payload.decided_by or principal.user_id),
            decided_at=datetime.now(timezone.utc)), execution_options={"synchronize_session": False})
    db.commit()
    if changed.rowcount != 1:
        raise HTTPException(status_code=409, detail="Approval already decided")
    db.refresh(approval)
    if not payload.approve:
        approval.execution_status = "not_executed"
        draft = db.query(MaintenanceDraftRecord).filter_by(approval_id=approval.id).one_or_none()
        if draft:
            draft.status = "REJECTED"
            draft.updated_at = datetime.now(timezone.utc)
        db.commit()
        AuditLogger(db, principal).log(approval.run_id or f"approval:{approval.id}", "approval_rejected", f"Rejected {approval.tool}", {"approval_id": approval.id, "decided_by": approval.decided_by})
        if draft:
            AuditLogger(db, principal).log(f"asset:{draft.asset_id}", "MAINTENANCE_DRAFT_REJECTED", "Maintenance draft rejected; no plant action was executed.", {"approval_id": approval.id, "draft_id": draft.id})
        return {"id": approval.id, "status": approval.status, "execution_status": approval.execution_status}

    settings = get_settings()
    revalidated = ActionGuard(settings.tools_config).evaluate(approval.tool, approved=True)
    args = json.loads(approval.args_json)
    current_hash = ContentIdentityService().hash_json({"tool": approval.tool, "arguments": args})
    result_payload: dict[str, object]
    requester = (
        LocalIdentityProvider(db, settings.access_config).principal_for_user(approval.requester_id)
        if approval.requester_id else principal
    )
    if requester is None and settings.auth_mode.lower() != "local":
        requester = principal
    tool_access = AuthorizationService().can_use_tool(requester, approval.tool) if requester else None
    requester_failure = None
    if requester and saved_state:
        try:
            saved_state = AgentRunState.model_validate_json(db.get(AgentRunRecord, saved_state.id, populate_existing=True).state_json)
            revalidate_principal(db, settings, requester, saved_state)
            if saved_state.status != RunStatus.waiting_for_approval:
                raise HTTPException(status_code=409, detail={"code": "TASK_NOT_WAITING_FOR_APPROVAL"})
        except HTTPException as exc:
            requester_failure = exc.detail
        tool_access = AuthorizationService().can_use_tool(requester, approval.tool)
    if approval.action_hash and current_hash != approval.action_hash:
        approval.status = "blocked"
        approval.execution_status = "blocked"
        result_payload = {"success": False, "error": "APPROVAL_ARGUMENTS_CHANGED"}
    elif requester_failure or not tool_access or not tool_access.allowed:
        approval.status = "blocked"
        approval.execution_status = "blocked"
        result_payload = {"success": False, "error": requester_failure or (tool_access.reason_code if tool_access else "REQUESTER_IDENTITY_INVALID")}
    elif revalidated.decision.value != "ALLOW":
        approval.status = "blocked"
        approval.execution_status = "blocked"
        result_payload = {"success": False, "error": revalidated.reason}
    elif approval.tool == "approve_maintenance_draft":
        draft = db.get(MaintenanceDraftRecord, str(args.get("draft_id", "")))
        if not draft or draft.approval_id != approval.id:
            approval.status = "blocked"
            approval.execution_status = "blocked"
            result_payload = {"success": False, "error": "MAINTENANCE_DRAFT_FAILED"}
        else:
            draft.status = "APPROVED"
            draft.updated_at = datetime.now(timezone.utc)
            approval.execution_status = "draft_accepted"
            result_payload = {
                "success": True, "draft_id": draft.id, "status": draft.status,
                "plant_action_executed": False,
            }
            AuditLogger(db, principal).log(f"asset:{draft.asset_id}", "MAINTENANCE_DRAFT_APPROVED", "Maintenance draft approved; no plant action was executed.", {"approval_id": approval.id, "draft_id": draft.id})
    else:
        registry = create_agent_registry(settings, db, requester, task_scope(saved_state)) if saved_state else create_default_registry(settings.workspace_root)
        try:
            errors = registry.validate_arguments(approval.tool, args)
            if errors:
                raise ValueError("; ".join(errors))
            record = next((item for item in saved_state.tool_records if item.get("approval_id") == approval.id), None) if saved_state else None
            if saved_state and (not record or record.get("arguments") != args or record.get("tool") != approval.tool):
                raise ValueError("APPROVAL_TASK_ACTION_MISMATCH")
            context = ActionContext(db, saved_state.id, record["operation_id"], approval.tool, current_hash, settings.workspace_root) if record else None
            with action_context(context):
                timeout = min(120, max(2, float(saved_state.runtime_metrics.get("execution_limits", {}).get("tool_timeout", 30)))) if saved_state else 30
                result = await asyncio.wait_for(registry.get(approval.tool).execute(args), timeout=timeout)
                persist_receipt(result)
            result_payload = result.model_dump(mode="json")
            approval.status = "executed" if result.success else "execution_failed"
            approval.execution_status = approval.status
        except asyncio.TimeoutError:
            approval.status, approval.execution_status = "approved", "unknown"
            result_payload = {"success": False, "error": "TOOL_OUTCOME_UNKNOWN"}
        except (KeyError, ValueError, OSError) as exc:
            result_payload = {"success": False, "error": str(exc)}
            approval.status = "blocked"
            approval.execution_status = "blocked"
    approval.executed_at = datetime.now(timezone.utc)
    approval.result_json = json.dumps(result_payload, ensure_ascii=False, default=str)
    db.commit()
    AuditLogger(db, principal).log(
        approval.run_id or f"approval:{approval.id}", "approval_execution",
        f"Approval {approval.id} finished with {approval.execution_status}",
        {"approval_id": approval.id, "tool": approval.tool, "arguments": args, "result": result_payload, "revalidation": revalidated.model_dump(mode="json")},
    )
    response = {"id": approval.id, "status": approval.status, "execution_status": approval.execution_status, "result": result_payload}
    if approval.status == "executed" and saved_state and saved_state.status == RunStatus.waiting_for_approval:
        from app.api.tasks import schedule_resume, _load_task_record
        try:
            current = AgentRunState.model_validate_json(_load_task_record(db, requester, saved_state.id).state_json)
            if current.status == RunStatus.waiting_for_approval:
                continuation = await schedule_resume(db, requester, current, current.checkpoint.revision)
                response["continuation"] = continuation.model_dump()
        except HTTPException as exc:
            response["continuation"] = {"status": "blocked", "detail": exc.detail}
            db.rollback()
    return response


@router.get("/approvals/{approval_id}")
async def get_approval(
    approval_id: str,
    principal: Principal = Depends(require_permission(Permission.task_read)),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    approval = db.get(HumanApprovalRecord, approval_id)
    if not approval:
        raise HTTPException(status_code=404, detail="Approval not found")
    if get_settings().auth_mode.lower() == "local" and not (
        approval.requester_id == principal.user_id
        or principal.has_permission(Permission.approval_approve)
        or principal.has_permission(Permission.audit_read)
    ):
        raise HTTPException(status_code=404, detail="Approval not found")
    return {
        "id": approval.id, "run_id": approval.run_id, "tool": approval.tool,
        "arguments": json.loads(approval.args_json), "risk": approval.risk,
        "status": approval.status, "execution_status": approval.execution_status,
        "result": json.loads(approval.result_json) if approval.result_json else None,
    }


@router.get("/audit/{run_id}")
async def get_audit(
    run_id: str,
    principal: Principal = Depends(require_permission(Permission.audit_read)),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    events = db.query(AuditEventRecord).filter_by(run_id=run_id).order_by(AuditEventRecord.created_at).all()
    return {"run_id": run_id, "events": [{"id": event.id, "type": event.event_type, "summary": event.summary, "payload": json.loads(event.payload_json), "timestamp": event.created_at.isoformat()} for event in events]}
