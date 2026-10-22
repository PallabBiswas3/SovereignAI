"""Validate and reconcile a saved task before any continuation is claimed."""
from __future__ import annotations

from hashlib import sha256
from datetime import timezone
import json

from fastapi import HTTPException
from sqlalchemy import select

from app.agent.state import AgentRunState, RunStatus, StepStatus
from app.agent.receipts import file_proof
from app.agent.tool_agent import BoundedToolAgent
from app.core.database import AgentRunRecord, TaskCheckpointRecord, HumanApprovalRecord, ToolReceiptRecord, ArtifactRecord, KnowledgeChunkRecord, KnowledgeDocument, TaskExecutionRecord
from app.governance.action_guard import ActionGuard
from app.governance.policy_engine import GovernanceDecision
from app.identity import ContentIdentityService
from app.identity.authorization import AuthorizationService
from app.identity.models import Permission, ResourceScope, ClearanceLevel
from app.identity.provider import LocalIdentityProvider
from app.orchestration.structured_plan import StructuredPlan, validate_plan, resolve
from app.orchestration.workflow_selector import WorkflowSelection, WorkflowHandler, WorkflowStrategy


def restore_plan(state, tools, max_calls):
    entries = state.structured_plan.get("steps", [])
    plan = StructuredPlan.model_validate({"action": "plan", "reason_summary": "Continue the saved validated plan.",
        "steps": [{key: value for key, value in entry.items() if key in {"id", "tool", "arguments", "depends_on"}} for entry in entries]})
    ordered = validate_plan(plan, tools, remaining_calls=max_calls)
    outputs, observations, signatures = {}, [], set()
    remaining = []
    for step in ordered:
        entry = next(item for item in entries if item["id"] == step.id)
        if entry["status"] != "completed":
            remaining.append(step)
            continue
        matches = [record for record in state.tool_records if record.get("plan_step_id") == step.id and record.get("success")]
        if len(matches) != 1 or "output" not in matches[0] or matches[0]["tool"] != step.tool:
            raise ValueError("Completed step is missing a unique confirmed output")
        args = resolve(step.arguments, outputs)
        actual = dict(matches[0]["arguments"])
        if step.tool.startswith("generate_"):
            if actual.pop("run_id", None) != state.id:
                raise ValueError("Completed artifact belongs to another task")
        if actual != args:
            raise ValueError("Completed step arguments do not match the saved dependency bindings")
        outputs[step.id] = matches[0]["output"]
        observations.append({"step_id": step.id, "tool": step.tool, "success": True, "output": matches[0]["output"]})
        signatures.add(json.dumps({"tool": step.tool, "arguments": args}, sort_keys=True))
    if remaining:
        plan = plan.model_copy(update={"steps": remaining})
        remaining = validate_plan(plan, tools, completed=set(outputs), remaining_calls=max_calls - len(outputs))
    return plan, remaining, outputs, observations, signatures


def deny(code: str, status: int = 409):
    raise HTTPException(status_code=status, detail={"code": code})


def task_scope(state):
    if not state.organization_id or not state.workspace_id or not state.principal_id:
        deny("RESUME_ORIGINAL_SCOPE_REQUIRED", 403)
    return ResourceScope(resource_id=state.id, organization_id=state.organization_id,
        workspace_id=state.workspace_id, department_id=state.department_id,
        owner_id=state.principal_id, classification=ClearanceLevel.parse(state.classification))


def revalidate_principal(db, settings, principal, state):
    db.expire_all()
    current = LocalIdentityProvider(db, settings.access_config).principal_for_user(principal.user_id) if settings.auth_mode.lower() == "local" else principal
    if current is None:
        deny("RESUME_IDENTITY_INVALID", 403)
    if current.user_id != state.principal_id:
        deny("RESUME_OWNER_REQUIRED", 403)
    authorization = AuthorizationService()
    for permission in (Permission.task_read, Permission.task_create):
        if not authorization.authorize(current, permission, task_scope(state)).allowed:
            deny("RESUME_ACCESS_REVOKED", 403)
    guard = ActionGuard(settings.tools_config)
    for record in state.tool_records:
        if not guard.tools.get(record["tool"], {}).get("enabled", False):
            deny("RESUME_TOOL_DISABLED", 403)
    from app.governance.policy_engine import PolicyEngine
    from app.governance.injection import PromptInjectionScanner
    from app.governance.pii import PIIDetector
    execution = db.get(TaskExecutionRecord, state.id)
    if execution:
        payload = json.loads(execution.payload_json)
        engine = PolicyEngine(settings.policies_config)
        if engine.decide(engine.get(payload.get("use_case", "internal_assistant")),
                         pii_count=len(PIIDetector().detect(state.request)),
                         injection_count=len(PromptInjectionScanner().scan(state.request))) == GovernanceDecision.block:
            deny("RESUME_INPUT_POLICY_BLOCKED", 403)
    # Registries and executors share this object. Refresh it before each resumed
    # action so a permission change cannot be hidden by an old registry closure.
    for name in ("permissions", "roles", "organization_id", "workspace_ids", "department_ids", "clearance"):
        setattr(principal, name, getattr(current, name))
    revalidate_evidence(db, settings, principal, state)
    from app.agent.input_replies import verify_reply_files
    verify_reply_files(settings, state)
    return principal


def revalidate_evidence(db, settings, principal, state):
    """Cached observations are evidence, not a grant of continuing access."""
    from app.assets.repository import AssetRepository
    from app.rag.retrieval import _scope_allows
    from app.api.artifacts import _scope as artifact_scope
    auth = AuthorizationService()
    for record in state.tool_records:
        authorize_tool(principal, record["tool"])
        output = record.get("output") or {}
        if record["tool"] == "knowledge_search" and isinstance(output, dict):
            if not principal.has_permission(Permission.knowledge_read):
                deny("RESUME_EVIDENCE_ACCESS_REVOKED", 403)
            for item in output.get("results", []):
                chunk = db.get(KnowledgeChunkRecord, item.get("chunk_id"), populate_existing=True)
                document = db.get(KnowledgeDocument, chunk.document_id, populate_existing=True) if chunk else None
                if not document or chunk.text != item.get("text"):
                    deny("RESUME_EVIDENCE_CHANGED")
                metadata = {**json.loads(document.metadata_json or "{}"), **json.loads(chunk.metadata_json or "{}")}
                if settings.auth_mode.lower() == "local" and not _scope_allows(metadata, ["internal"], principal):
                    deny("RESUME_EVIDENCE_ACCESS_REVOKED", 403)
        if record["tool"] in {"get_asset_telemetry", "get_asset_history"} and isinstance(output, dict) and output:
            repository = AssetRepository(db)
            row = repository.get_row(str(output.get("asset_id", "")))
            if not row or not repository.is_authorized(principal, row, telemetry=True):
                deny("RESUME_EVIDENCE_ACCESS_REVOKED", 403)
    for item in state.artifacts:
        artifact = db.get(ArtifactRecord, item["id"], populate_existing=True)
        scope = artifact_scope(artifact) if artifact else None
        if not artifact or artifact.run_id != state.id or not scope or not auth.can_read_artifact(principal, scope).allowed:
            deny("RESUME_ARTIFACT_ACCESS_REVOKED", 403)


def authorize_tool(principal, tool):
    if not AuthorizationService().can_use_tool(principal, tool).allowed:
        deny("RESUME_TOOL_ACCESS_REVOKED", 403)
    permissions = {
        "knowledge_search": [Permission.knowledge_read],
        "get_asset_telemetry": [Permission.asset_read, Permission.telemetry_read],
        "get_asset_history": [Permission.asset_read, Permission.telemetry_read],
        "generate_docx": [Permission.artifact_create],
        "generate_xlsx": [Permission.artifact_create],
        "generate_pptx": [Permission.artifact_create],
    }
    if any(not principal.has_permission(permission) for permission in permissions.get(tool, [])):
        deny("RESUME_TOOL_ACCESS_REVOKED", 403)


def _verify_files(db, settings, state, principal, receipt):
    for proof in json.loads(receipt.files_json):
        root = settings.workspace_root
        if proof["kind"] == "artifact":
            from app.api.artifacts import _scope as artifact_scope
            artifact = db.get(ArtifactRecord, proof["artifact_id"])
            if not artifact or artifact.run_id != state.id or artifact.sha256 != proof["sha256"] or artifact.path != proof["path"] or artifact.size != proof["size"]:
                deny("RESUME_ARTIFACT_RECEIPT_MISMATCH")
            scope = artifact_scope(artifact)
            if not scope or not AuthorizationService().can_read_artifact(principal, scope).allowed:
                deny("RESUME_ARTIFACT_ACCESS_REVOKED", 403)
            root = root / "artifacts"
        try:
            actual = file_proof(root, proof["path"])
        except (OSError, ValueError):
            deny("RESUME_OUTPUT_UNAVAILABLE")
        if actual["sha256"] != proof["sha256"] or actual["size"] != proof["size"]:
            deny("RESUME_OUTPUT_CHANGED")


def prepare_resume(db, settings, state, execution, principal, expected_revision, tools, *, input_reply=None):
    revalidate_principal(db, settings, principal, state)
    if state.checkpoint is None or execution is None:
        deny("RESUME_CHECKPOINT_REQUIRED")
    if state.checkpoint.schema_version != 1:
        deny("RESUME_CHECKPOINT_VERSION_UNSUPPORTED")
    if execution.phase not in {"paused", "interrupted", "finished"}:
        deny("RESUME_TASK_IS_ACTIVE")
    if state.status in {RunStatus.completed, RunStatus.blocked} or (state.status == RunStatus.waiting_for_input and input_reply is None):
        deny("RESUME_TASK_NOT_ELIGIBLE")
    if execution.revision != expected_revision or state.checkpoint.revision != expected_revision:
        deny("RESUME_STALE_REVISION")
    latest = db.scalar(select(TaskCheckpointRecord).where(TaskCheckpointRecord.run_id == state.id,
                                                        TaskCheckpointRecord.revision == expected_revision))
    row_raw = db.get(AgentRunRecord, state.id).state_json
    if latest is None or sha256(row_raw.encode()).hexdigest() != latest.state_sha256:
        deny("RESUME_CHECKPOINT_INTEGRITY_FAILED")
    payload = json.loads(execution.payload_json)
    expected_payload_hash = state.checkpoint.data.get("request_payload_sha256")
    if expected_payload_hash and expected_payload_hash != sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest():
        deny("RESUME_PAYLOAD_CHANGED")
    if payload.get("request") != state.request:
        deny("RESUME_REQUEST_CHANGED")
    resumed = state.model_copy(deep=True)
    if input_reply is not None:
        from app.agent.input_replies import apply_reply
        payload = apply_reply(db, settings, resumed, payload, principal, input_reply)
    if int(resumed.checkpoint.data.get("resume_attempts", 0)) >= 3:
        deny("RESUME_ATTEMPT_LIMIT")
    budget = resumed.runtime_metrics.get("execution_budget", resumed.runtime_metrics.get("execution_limits", {}))
    if float(resumed.runtime_metrics.get("active_seconds", 0)) >= float(budget.get("max_seconds", 180)):
        deny("RESUME_TIME_BUDGET_EXHAUSTED")
    if int(resumed.runtime_metrics.get("decisions_used", 0)) >= int(budget.get("max_decisions", 8)):
        deny("RESUME_DECISION_BUDGET_EXHAUSTED")
    if resumed.checkpoint.data.get("active_operations"):
        deny("RESUME_COMPOSITE_OUTCOME_UNRESOLVED")
    guard, identity = ActionGuard(settings.tools_config), ContentIdentityService()
    for record in resumed.tool_records:
        tool = str(record["tool"])
        if not AuthorizationService().can_use_tool(principal, tool).allowed:
            deny("RESUME_TOOL_ACCESS_REVOKED", 403)
        if not guard.tools.get(tool, {}).get("enabled", False):
            deny("RESUME_TOOL_DISABLED", 403)
        try:
            tools.get(tool)
        except KeyError:
            deny("RESUME_TOOL_OUTSIDE_SCOPE", 403)
        action_hash = identity.hash_json({"tool": tool, "arguments": record["arguments"]})
        if record.get("action_hash") and record["action_hash"] != action_hash:
            deny("RESUME_ACTION_CHANGED")
        if tool in {"knowledge_search", "get_asset_telemetry", "get_asset_history"}:
            needed = {"knowledge_search": Permission.knowledge_read,
                      "get_asset_telemetry": Permission.telemetry_read,
                      "get_asset_history": Permission.telemetry_read}[tool]
            if not principal.has_permission(needed):
                deny("RESUME_EVIDENCE_ACCESS_REVOKED", 403)
        if tool in {"read_file", "get_file_metadata", "extract_pdf_text", "ocr_document", "analyze_image"} and not record.get("waiting_for_approval"):
            if not record.get("input_proofs"):
                deny("RESUME_INPUT_FINGERPRINT_REQUIRED")
            for proof in record["input_proofs"]:
                try:
                    actual = file_proof(settings.workspace_root, proof["path"])
                except (OSError, ValueError):
                    deny("RESUME_INPUT_UNAVAILABLE")
                if actual != proof:
                    deny("RESUME_INPUT_CHANGED")
        receipt = db.get(ToolReceiptRecord, record.get("operation_id", ""))
        result, evidence = None, None
        if receipt:
            if receipt.run_id != state.id or receipt.tool != tool or receipt.action_hash != action_hash:
                deny("RESUME_RECEIPT_MISMATCH")
            if sha256(receipt.result_json.encode()).hexdigest() != receipt.result_sha256:
                deny("RESUME_RECEIPT_INTEGRITY_FAILED")
            _verify_files(db, settings, state, principal, receipt)
            result, evidence = json.loads(receipt.result_json), {"type": "tool_receipt", "id": receipt.operation_id}
        if record.get("waiting_for_approval"):
            approval = db.get(HumanApprovalRecord, record.get("approval_id", ""))
            if not approval or approval.run_id != state.id or approval.requester_id != state.principal_id or approval.tool != tool or approval.action_hash != action_hash:
                deny("RESUME_APPROVAL_MISMATCH")
            if approval.organization_id != state.organization_id or approval.workspace_id != state.workspace_id:
                deny("RESUME_APPROVAL_SCOPE_CHANGED", 403)
            if guard.evaluate(tool, approved=True).decision != GovernanceDecision.allow:
                deny("RESUME_APPROVAL_POLICY_CHANGED", 403)
            if approval.status == "pending":
                deny("RESUME_APPROVAL_PENDING")
            if approval.status in {"blocked", "rejected", "execution_failed"}:
                deny("RESUME_APPROVAL_NOT_EXECUTED")
            if approval.status == "executed" and approval.result_json:
                approved_result = json.loads(approval.result_json)
                if not approved_result.get("success"):
                    deny("RESUME_APPROVAL_FAILED")
                if result and result != approved_result:
                    deny("RESUME_APPROVAL_RECEIPT_MISMATCH")
                result, evidence = approved_result, {"type": "approval", "id": approval.id}
            elif not result:
                deny("RESUME_APPROVAL_OUTCOME_UNKNOWN")
            elif approval.status != "approved" or approval.execution_status not in {"executing", "unknown"}:
                deny("RESUME_APPROVAL_STATE_INVALID")
            else:
                evidence["approval_id"] = approval.id
            if result and (tool.startswith("generate_") or tool in {"write_file", "run_python"}) and not receipt:
                deny("RESUME_SIDE_EFFECT_RECEIPT_REQUIRED")
            finished_at = approval.executed_at or (receipt.created_at if receipt else None)
            if result and approval.decided_at and finished_at:
                active = max(0, (finished_at.replace(tzinfo=timezone.utc) - approval.decided_at.replace(tzinfo=timezone.utc)).total_seconds())
                resumed.runtime_metrics["active_seconds"] = float(resumed.runtime_metrics.get("active_seconds", 0)) + active
        if result:
            if not result.get("success"):
                deny("RESUME_RECEIPT_FAILED")
            if record.get("success") is True and record.get("output") != result.get("output"):
                deny("RESUME_SAVED_OUTPUT_CHANGED")
            record.update(success=True, output=result.get("output"), execution_status="succeeded",
                          error=None, reconciliation=evidence, waiting_for_approval=False,
                          output_summary=BoundedToolAgent._summarize(result.get("output")))
            BoundedToolAgent._collect_result(resumed, result.get("output"))
        elif record.get("success") is not True:
            deny("RESUME_ACTION_OUTCOME_UNKNOWN")
        elif tool.startswith("generate_") or tool in {"write_file", "run_python"}:
            deny("RESUME_SIDE_EFFECT_RECEIPT_REQUIRED")
        for step in resumed.plan.steps:
            if step.action == tool and step.args == record["arguments"] and (not record.get("agent_step_id") or step.id == record["agent_step_id"]) and step.status in {StepStatus.waiting_for_approval, StepStatus.interrupted, StepStatus.cancelled, StepStatus.running}:
                step.status = StepStatus.completed
                step.error = None
                step.observation = "Completed output verified against its durable execution receipt."
        if record.get("plan_step_id"):
            for step in resumed.structured_plan.get("steps", []):
                if step["id"] == record["plan_step_id"]:
                    step["status"] = "completed"
    if float(resumed.runtime_metrics.get("active_seconds", 0)) >= float(budget.get("max_seconds", 180)):
        deny("RESUME_TIME_BUDGET_EXHAUSTED")
    if not resumed.workflow_selection and resumed.checkpoint.data.get("workflow_selection"):
        resumed.workflow_selection = WorkflowSelection.model_validate(resumed.checkpoint.data["workflow_selection"])
    handler = (resumed.workflow_selection.model_dump(mode="json") if resumed.workflow_selection else
               resumed.checkpoint.data.get("workflow_selection", {})).get("handler")
    if resumed.checkpoint.data.get("input_continuation"):
        handler = resumed.checkpoint.data["input_continuation"]
        if handler == "diagnostic_input" and (resumed.checkpoint.data.get("operation_outputs") or resumed.execution_records or resumed.tool_records):
            deny("RESUME_HANDLER_REQUIRES_VALIDATED_CONTINUATION")
    if resumed.structured_plan.get("status") == "fallback":
        handler = "tool_agent"
        if resumed.workflow_selection:
            resumed.workflow_selection.handler = WorkflowHandler.tools
            resumed.workflow_selection.strategy = WorkflowStrategy.adaptive
            resumed.workflow_selection.fallback_reason = resumed.runtime_metrics.get("workflow_fallback")
    if handler is None and not (resumed.plan.steps or resumed.tool_records or resumed.execution_records
                               or resumed.checkpoint.data.get("operation_outputs") or resumed.workcell_state):
        handler = "select_workflow"
    entries = resumed.structured_plan.get("steps") if resumed.structured_plan.get("status") != "fallback" else None
    all_done = all(step.get("status") == "completed" for step in entries) if entries else bool(resumed.plan.steps) and all(step.status == StepStatus.completed for step in resumed.plan.steps)
    draft_ready = bool(resumed.checkpoint.data.get("draft_ready"))
    if resumed.status == RunStatus.waiting_for_approval and not any(item.get("approval_id") for item in resumed.tool_records):
        deny("RESUME_RESPONSE_REVIEW_REQUIRED")
    if handler not in {"tool_agent", "structured_plan", "direct_answer", "select_workflow", "diagnostic_input"}:
        deny("RESUME_HANDLER_REQUIRES_VALIDATED_CONTINUATION")
    if handler == "structured_plan" and resumed.structured_plan.get("steps"):
        try:
            _, remaining, _, _, _ = restore_plan(resumed, tools, min(6, int(budget.get("max_tool_calls", 6))))
            for step in remaining:
                authorize_tool(principal, step.tool)
                if not AuthorizationService().can_use_tool(principal, step.tool).allowed or guard.evaluate(step.tool).decision == GovernanceDecision.block:
                    deny("RESUME_REMAINING_TOOL_DENIED", 403)
        except (ValueError, TypeError, KeyError):
            deny("RESUME_PLAN_NO_LONGER_VALID")
    if all_done and draft_ready:
        handler = "finalize_only"
    # Keep the original evidence and results; clear only the interrupted outcome.
    resumed.status, resumed.lifecycle = RunStatus.running, None
    resumed.checkpoint.interruption_reason = None
    resumed.checkpoint.data["resume_attempts"] = int(resumed.checkpoint.data.get("resume_attempts", 0)) + 1
    resumed.checkpoint.data["resume_from_revision"] = expected_revision
    resumed.checkpoint.data["resume_handler"] = handler
    resumed.checkpoint.data["resume_baseline_artifact_ids"] = [item["id"] for item in resumed.artifacts]
    if resumed.checkpoint.data.get("draft_response"):
        resumed.final_response = resumed.checkpoint.data["draft_response"]
    revalidate_evidence(db, settings, principal, resumed)
    return resumed, payload
