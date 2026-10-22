from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.router.schemas import RoutingDecision
from app.orchestration.workflow_selector import WorkflowSelection


class StepStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    waiting_for_approval = "waiting_for_approval"
    waiting_for_input = "waiting_for_input"
    cancelled = "cancelled"
    blocked = "blocked"
    interrupted = "interrupted"


class RunStatus(str, Enum):
    planning = "planning"
    running = "running"
    completed = "completed"
    failed = "failed"
    waiting_for_approval = "waiting_for_approval"
    waiting_for_input = "waiting_for_input"
    cancelled = "cancelled"
    blocked = "blocked"
    interrupted = "interrupted"


class ExecutionCheckpoint(BaseModel):
    schema_version: int = 1
    revision: int = 0
    boundary: str = "accepted"
    phase: Literal["pending", "active", "paused", "finished", "interrupted"] = "pending"
    saved_at: datetime | None = None
    requires_reconciliation: bool = False
    interruption_reason: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class RequiredTaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    prompt: str = Field(min_length=1, max_length=1000)
    kind: Literal["text", "attachment"] = "text"


class CompletionCheck(BaseModel):
    code: str
    passed: bool
    detail: str


class TaskLifecycle(BaseModel):
    """Execution outcome, distinct from the response's release decision.

    This is persisted task metadata, not a resumable execution checkpoint.
    Step IDs belong to step_source (structured_plan or agent_plan).
    """

    schema_version: int = 1
    status: RunStatus
    terminal: bool = False
    execution_complete: bool = False
    step_source: Literal["agent_plan", "structured_plan"] = "agent_plan"
    current_step_id: str | None = None
    completed_step_ids: list[str] = Field(default_factory=list)
    remaining_step_ids: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
    waiting_reason: str | None = None
    required_inputs: list[RequiredTaskInput] = Field(default_factory=list)
    approval_ids: list[str] = Field(default_factory=list)
    response_release: Literal["not_evaluated", "allowed", "allowed_with_warning", "needs_revision", "waiting_for_approval", "blocked"] = "not_evaluated"
    completion_checks: list[CompletionCheck] = Field(default_factory=list)


class AgentStep(BaseModel):
    id: int
    action: str
    title: str
    status: StepStatus = StepStatus.pending
    args: dict[str, Any] = Field(default_factory=dict)
    observation: str | None = None
    verification: str | None = None
    error: str | None = None


class AgentPlan(BaseModel):
    goal: str
    steps: list[AgentStep]


class AgentRunState(BaseModel):
    id: str
    request: str
    status: RunStatus
    plan: AgentPlan
    routing: RoutingDecision
    requested_execution_mode: str = "AUTOMATIC"
    execution_mode: str = "STANDARD"
    execution_mode_reason: str = "Default bounded execution mode."
    requested_chat_mode: str = "AUTOMATIC"
    chat_mode: str = "GENERAL"
    chat_mode_reason: str = "No organizational evidence or governed action was requested."
    runtime_metrics: dict[str, Any] = Field(default_factory=dict)
    workflow_selection: WorkflowSelection | None = None
    structured_plan: dict[str, Any] = Field(default_factory=dict)
    lifecycle: TaskLifecycle | None = None
    checkpoint: ExecutionCheckpoint | None = None
    final_response: str | None = None
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[dict[str, str]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    evidence_records: list[dict[str, Any]] = Field(default_factory=list)
    measurements: list[dict[str, Any]] = Field(default_factory=list)
    rules: list[dict[str, Any]] = Field(default_factory=list)
    calculations: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    asset_context: dict[str, Any] | None = None
    trend_analyses: list[dict[str, Any]] = Field(default_factory=list)
    maintenance_history: list[dict[str, Any]] = Field(default_factory=list)
    maintenance_draft: dict[str, Any] | None = None
    context_metrics: dict[str, Any] = Field(default_factory=dict)
    retrieval_metrics: list[dict[str, Any]] = Field(default_factory=list)
    governance: dict[str, Any] = Field(default_factory=dict)
    execution_records: list[dict[str, Any]] = Field(default_factory=list)
    tool_records: list[dict[str, Any]] = Field(default_factory=list)
    workcell_id: str | None = None
    workcell_version: str | None = None
    workcell_hash: str | None = None
    workflow_version: str | None = None
    workcell_state: dict[str, Any] = Field(default_factory=dict)
    capsule: dict[str, Any] | None = None
    principal_id: str | None = None
    organization_id: str | None = None
    workspace_id: str | None = None
    department_id: str | None = None
    classification: str = "INTERNAL"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
