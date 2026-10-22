"""Validate a reply against only the questions in the saved task checkpoint."""
from datetime import datetime, timezone
from hashlib import sha256
import json

from fastapi import HTTPException

from app.agent.lifecycle import assess_lifecycle
from app.agent.receipts import file_proof
from app.agent.state import RunStatus, StepStatus
from app.core.database import TaskInputUploadRecord
from app.governance.injection import PromptInjectionScanner
from app.governance.pii import PIIDetector
from app.governance.policy_engine import PolicyEngine, GovernanceDecision
from app.integrations.routing import IntegrationRoutePlanner


def invalid(code, status=422):
    raise HTTPException(status_code=status, detail={"code": code})


def apply_reply(db, settings, state, payload, principal, reply):
    if state.status != RunStatus.waiting_for_input or state.checkpoint.phase != "paused":
        invalid("TASK_NOT_WAITING_FOR_INPUT", 409)
    assess_lifecycle(state)
    questions = state.lifecycle.required_inputs
    if not questions or len(questions) > 3:
        invalid("TASK_INPUT_QUESTIONS_UNAVAILABLE", 409)
    answers, attachments = reply.answers, reply.attachments
    required_text = {item.key for item in questions if item.kind == "text"}
    required_files = {item.key for item in questions if item.kind == "attachment"}
    if set(answers) != required_text or set(attachments) != required_files:
        invalid("TASK_INPUT_FIELDS_MISMATCH")
    if any(not value.strip() for value in answers.values()):
        invalid("TASK_INPUT_EMPTY")
    supplied = "\n".join(answers.values())
    engine = PolicyEngine(settings.policies_config)
    if engine.decide(engine.get(payload.get("use_case", "internal_assistant")),
                     pii_count=len(PIIDetector().detect(supplied)),
                     injection_count=len(PromptInjectionScanner().scan(supplied))) == GovernanceDecision.block:
        invalid("TASK_INPUT_POLICY_BLOCKED", 403)
    proofs, paths = [], {}
    for key, upload_id in attachments.items():
        upload = db.get(TaskInputUploadRecord, upload_id)
        if not upload or upload.run_id != state.id or upload.owner_id != principal.user_id or upload.input_key != key:
            invalid("TASK_INPUT_UPLOAD_NOT_FOUND", 404)
        try:
            proof = file_proof(settings.workspace_root, upload.path)
        except (ValueError, OSError):
            invalid("TASK_INPUT_FILE_UNAVAILABLE")
        if proof["sha256"] != upload.sha256 or proof["size"] != upload.size:
            invalid("TASK_INPUT_FILE_CHANGED")
        proofs.append(proof)
        paths[key] = upload.path
    handler = (state.workflow_selection.handler.value if state.workflow_selection else
               state.checkpoint.data.get("workflow_selection", {}).get("handler"))
    diagnostic = handler == "industrial_integration" and state.context_metrics.get("integration_route", {}).get("missing_diagnostic_input")
    if diagnostic:
        if state.tool_records or state.checkpoint.data.get("active_operations") or state.checkpoint.data.get("operation_outputs"):
            invalid("TASK_INPUT_UNSAFE_CONTINUATION", 409)
        route = IntegrationRoutePlanner().plan(state.request, list(paths.values()), settings.workspace_root)
        original = state.context_metrics["integration_route"]
        if not route.diagnostic:
            invalid("TASK_INPUT_INVALID_DIAGNOSTIC_JSON")
        if route.diagnostic_domain != original.get("diagnostic_domain") or route.use_graph != original.get("use_graph"):
            invalid("TASK_INPUT_DIAGNOSTIC_SCOPE_CHANGED")
        state.context_metrics["integration_route"] = {**original, "missing_diagnostic_input": False}
        state.checkpoint.data["diagnostic_attachment"] = paths["diagnostic_input"]
        for step in state.plan.steps:
            if step.action == "diagnose_time_series" and step.status == StepStatus.waiting_for_input:
                step.status = StepStatus.pending
    elif handler not in {"tool_agent", "structured_plan"}:
        invalid("TASK_INPUT_HANDLER_UNSUPPORTED", 409)
    else:
        from app.identity.models import Permission
        if not principal.has_permission(Permission.tool_execute):
            invalid("RESUME_TOOL_ACCESS_REVOKED", 403)
    if len(state.checkpoint.data.get("input_history", [])) >= 3:
        invalid("TASK_INPUT_ROUND_LIMIT", 409)
    updated = {**payload, "attachments": list(dict.fromkeys(payload.get("attachments", []) + list(paths.values())))}
    if payload.get("tool_scope") == "NUMERIC" and paths:
        invalid("TASK_INPUT_ATTACHMENT_OUTSIDE_SCOPE")
    if len(updated["attachments"]) > 10:
        invalid("TASK_INPUT_ATTACHMENT_LIMIT")
    state.checkpoint.data.setdefault("original_payload", payload)
    state.checkpoint.data.setdefault("input_history", []).append({
        "from_revision": state.checkpoint.revision, "answers": answers, "attachments": paths,
        "file_proofs": proofs, "submitted_at": datetime.now(timezone.utc).isoformat(),
        "principal_id": principal.user_id,
        "questions": [item.model_dump(mode="json") for item in questions],
    })
    state.checkpoint.data["request_payload_sha256"] = sha256(json.dumps(updated, sort_keys=True).encode()).hexdigest()
    state.lifecycle.required_inputs = []
    for step in state.plan.steps:
        if step.action == "ask_user" and step.status == StepStatus.waiting_for_input:
            step.status = StepStatus.completed
            step.observation = "Requested user inputs were validated and retained in task history."
    state.checkpoint.data["input_continuation"] = "diagnostic_input" if diagnostic else handler
    return updated


def verify_reply_files(settings, state):
    for item in state.checkpoint.data.get("input_history", []) if state.checkpoint else []:
        for proof in item.get("file_proofs", []):
            try:
                current = file_proof(settings.workspace_root, proof["path"])
            except (OSError, ValueError):
                invalid("TASK_INPUT_FILE_UNAVAILABLE", 409)
            if current != proof:
                invalid("TASK_INPUT_FILE_CHANGED", 409)
