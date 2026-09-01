from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.router.schemas import RoutingDecision


class StepStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    waiting_for_approval = "waiting_for_approval"


class RunStatus(str, Enum):
    planning = "planning"
    running = "running"
    completed = "completed"
    failed = "failed"
    waiting_for_approval = "waiting_for_approval"


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
    runtime_metrics: dict[str, Any] = Field(default_factory=dict)
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
    context_metrics: dict[str, Any] = Field(default_factory=dict)
    retrieval_metrics: list[dict[str, Any]] = Field(default_factory=list)
    governance: dict[str, Any] = Field(default_factory=dict)
    execution_records: list[dict[str, Any]] = Field(default_factory=list)
    tool_records: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
