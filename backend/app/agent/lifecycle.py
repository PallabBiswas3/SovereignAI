"""Deterministic completion checks shared by persisted tasks and their API views."""
from __future__ import annotations

from datetime import datetime, timezone

from app.agent.state import AgentRunState, CompletionCheck, RequiredTaskInput, RunStatus, TaskLifecycle


TERMINAL_STATUSES = {RunStatus.completed, RunStatus.failed, RunStatus.cancelled, RunStatus.blocked}
PAUSED_STATUSES = {RunStatus.waiting_for_input, RunStatus.waiting_for_approval, RunStatus.interrupted}


def assess_lifecycle(state: AgentRunState, *, touch: bool = False) -> AgentRunState:
    """Reconcile handler claims with actual step outcomes without any model call.

    Also derives metadata for old saved runs, without modifying stored records on
    read. A successful execution does not itself authorize release of its answer.
    """
    previous = state.lifecycle
    lifecycle = TaskLifecycle(
        status=state.status,
        required_inputs=previous.required_inputs if previous else [],
    )
    structured = state.structured_plan
    if structured.get("steps") and structured.get("status") != "fallback":
        lifecycle.step_source = "structured_plan"
        steps = [(str(item["id"]), str(item.get("status", "pending"))) for item in structured["steps"]]
    else:
        steps = [(str(item.id), item.status.value) for item in state.plan.steps]
    lifecycle.completed_step_ids = [key for key, status in steps if status == "completed"]
    lifecycle.remaining_step_ids = [key for key, status in steps if status != "completed"]
    lifecycle.current_step_id = next((key for key, status in steps if status in {
        "running", "failed", "waiting_for_approval", "waiting_for_input", "cancelled", "blocked", "interrupted",
    }), next(iter(lifecycle.remaining_step_ids), None))
    waiting_actions = [item for item in state.tool_records if item.get("waiting_for_approval")]
    lifecycle.approval_ids = list(dict.fromkeys(
        str(item["approval_id"]) for item in waiting_actions if item.get("approval_id")
    ))
    route = state.context_metrics.get("integration_route") or {}
    if route.get("missing_diagnostic_input") and not lifecycle.required_inputs:
        lifecycle.required_inputs = [RequiredTaskInput(
            key="diagnostic_input", kind="attachment",
            prompt="Attach diagnostic JSON with contract_version, domain, inputs and run_context containing measured time-series data.",
        )]

    decision = str(state.governance.get("decision") or "")
    controlplane = str(state.context_metrics.get("controlplane_status") or "")
    workflow_blocked = bool(state.workflow_selection and state.workflow_selection.strategy.value == "BLOCKED")
    lifecycle.response_release = {
        "ALLOW": "allowed", "ALLOW_WITH_WARNING": "allowed_with_warning",
        "REWRITE": "needs_revision", "REQUIRE_HUMAN_APPROVAL": "waiting_for_approval",
        "BLOCK": "blocked",
    }.get(decision, "not_evaluated")
    if controlplane and controlplane != "released":
        lifecycle.response_release = "waiting_for_approval" if "human_review" in controlplane else "blocked"

    checks = [
        CompletionCheck(code="all_steps_completed", passed=all(status == "completed" for _, status in steps),
                        detail="Every step in the active execution plan must have completed."),
        CompletionCheck(code="response_present", passed=bool((state.final_response or "").strip()),
                        detail="A completed task must retain a non-empty response."),
        CompletionCheck(code="tool_results_successful", passed=all(item.get("success") is True for item in state.tool_records),
                        detail="Recorded tool calls must have successful results before the task can complete."),
        CompletionCheck(code="required_inputs_present", passed=not lifecycle.required_inputs and state.status != RunStatus.waiting_for_input
                        and not any(status == "waiting_for_input" for _, status in steps),
                        detail="Required input must be supplied before execution can complete."),
        CompletionCheck(code="no_pending_approval", passed=not waiting_actions and state.status != RunStatus.waiting_for_approval
                        and not any(status == "waiting_for_approval" for _, status in steps),
                        detail="Pending actions and release reviews cannot count as completed work."),
    ]
    lifecycle.completion_checks = checks

    if state.status not in {RunStatus.failed, RunStatus.cancelled, RunStatus.interrupted}:
        if state.status == RunStatus.blocked or workflow_blocked or state.governance.get("input_decision") == "BLOCK":
            state.status = RunStatus.blocked
        elif state.status == RunStatus.completed and any(status == "failed" for _, status in steps):
            state.status = RunStatus.failed
        elif lifecycle.required_inputs or any(status == "waiting_for_input" for _, status in steps):
            state.status = RunStatus.waiting_for_input
        elif waiting_actions or state.status == RunStatus.waiting_for_approval or any(status == "waiting_for_approval" for _, status in steps):
            state.status = RunStatus.waiting_for_approval
        elif state.status == RunStatus.completed and not all(check.passed for check in checks):
            state.status = RunStatus.failed

    lifecycle.status = state.status
    checks.append(CompletionCheck(
        code="executor_finished_successfully", passed=state.status == RunStatus.completed,
        detail="The executor must finish successfully rather than stop, pause, or be blocked.",
    ))
    if state.status == RunStatus.waiting_for_approval:
        lifecycle.response_release = "waiting_for_approval"
    elif state.status in {RunStatus.waiting_for_input, RunStatus.interrupted}:
        lifecycle.response_release = "not_evaluated"
    lifecycle.terminal = state.status in TERMINAL_STATUSES
    lifecycle.execution_complete = state.status == RunStatus.completed and all(check.passed for check in checks)
    if state.status in {RunStatus.failed, RunStatus.cancelled, RunStatus.blocked}:
        lifecycle.failure_reason = (
            (previous.failure_reason if previous else None)
            or next((step.error for step in reversed(state.plan.steps) if step.error), None)
            or next((str(record["error"]) for record in reversed(state.tool_records) if record.get("error")), None)
            or (state.warnings[-1] if state.warnings and state.status == RunStatus.failed else None)
            or ("Task execution was cancelled." if state.status == RunStatus.cancelled else None)
            or (state.final_response if state.status == RunStatus.blocked else None)
            or "Task did not satisfy its execution completion checks."
        )
    elif state.status in PAUSED_STATUSES:
        lifecycle.waiting_reason = (
            (state.checkpoint.interruption_reason if state.status == RunStatus.interrupted and state.checkpoint else None)
            or
            (previous.waiting_reason if previous else None)
            or (lifecycle.required_inputs[0].prompt if lifecycle.required_inputs else None)
            or (state.final_response if state.status == RunStatus.waiting_for_approval else None)
            or ("Required task input is missing." if state.status == RunStatus.waiting_for_input else None)
            or "Human approval is required before continuing."
        )
    state.lifecycle = lifecycle
    if touch:
        state.updated_at = datetime.now(timezone.utc)
    return state


def outcome_event(state: AgentRunState) -> str:
    """End an execution stream honestly; a paused task has not completed."""
    return {
        RunStatus.completed: "task_completed", RunStatus.failed: "task_failed",
        RunStatus.cancelled: "task_cancelled", RunStatus.blocked: "task_blocked",
        RunStatus.waiting_for_approval: "task_paused", RunStatus.waiting_for_input: "task_paused",
        RunStatus.interrupted: "task_interrupted",
    }.get(state.status, "task_paused")
