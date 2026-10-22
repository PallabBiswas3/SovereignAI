"""Durable execution snapshots. This module never resumes or replays an action."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import time
from typing import Any
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.agent.lifecycle import assess_lifecycle, PAUSED_STATUSES
from app.agent.state import AgentRunState, ExecutionCheckpoint, RunStatus, StepStatus
from app.core.database import AgentRunRecord, AuditEventRecord, TaskCheckpointRecord, TaskExecutionRecord
from app.identity.authorization import AuthorizationService
from app.identity.models import Principal, ResourceScope, ClearanceLevel
from app.llm.base import ModelGenerationCancelled


LEASE_SECONDS = 60
HEARTBEAT_SECONDS = 10
_writer: ContextVar[CheckpointWriter | None] = ContextVar("task_checkpoint_writer", default=None)


def execution_id() -> str:
    writer = _writer.get()
    return writer.state.id if writer else str(uuid4())


def execution_scope(principal: Principal) -> ResourceScope:
    writer = _writer.get()
    if writer:
        state = writer.state
        return ResourceScope(resource_id=state.id, organization_id=state.organization_id,
            workspace_id=state.workspace_id, department_id=state.department_id,
            owner_id=state.principal_id, classification=ClearanceLevel.parse(state.classification))
    return AuthorizationService.owned_scope(principal)


def _scope(state: AgentRunState, principal: Principal) -> None:
    if state.organization_id and state.workspace_id and state.principal_id:
        return  # Resuming must not move a task into the user's first current workspace.
    scope = AuthorizationService.owned_scope(principal)
    state.principal_id = principal.user_id
    state.organization_id = scope.organization_id
    state.workspace_id = scope.workspace_id
    state.department_id = scope.department_id
    state.classification = scope.classification.name.upper()


def _append(db: Session, state: AgentRunState, execution: TaskExecutionRecord,
            boundary: str, phase: str) -> None:
    now = datetime.now(timezone.utc)
    state.checkpoint = state.checkpoint or ExecutionCheckpoint()
    execution.revision += 1
    execution.phase = phase
    state.checkpoint.revision = execution.revision
    state.checkpoint.boundary = boundary
    state.checkpoint.phase = phase
    state.checkpoint.saved_at = now
    state.checkpoint.requires_reconciliation = any(
        item.get("execution_status") in {"in_flight", "unknown"}
        for item in state.tool_records
    ) or bool(state.checkpoint.data.get("active_operations"))
    state.updated_at = now
    # Draft generation finishing is not the final governance/release boundary.
    snapshot = state.model_copy(deep=True)
    if phase == "active" and snapshot.status == RunStatus.completed:
        snapshot.status = RunStatus.running
    assess_lifecycle(snapshot)
    raw = snapshot.model_dump_json()
    row = db.get(AgentRunRecord, state.id)
    if row is None:
        row = AgentRunRecord(id=state.id, request=state.request, created_at=state.created_at)
        db.add(row)
    row.status = snapshot.status.value
    row.state_json = raw
    row.updated_at = now
    row.organization_id = state.organization_id
    row.owner_id = state.principal_id
    row.workspace_id = state.workspace_id
    row.department_id = state.department_id
    row.classification = state.classification
    db.add(TaskCheckpointRecord(
        id=str(uuid4()), run_id=state.id, revision=execution.revision, boundary=boundary,
        phase=phase, state_json=raw, state_sha256=sha256(raw.encode()).hexdigest(), created_at=now,
    ))
    db.add(AuditEventRecord(
        id=str(uuid4()), run_id=state.id, event_type="task_checkpoint",
        summary=f"Checkpoint {execution.revision}: {boundary}",
        payload_json=json.dumps({"revision": execution.revision, "boundary": boundary, "phase": phase,
                                 "requires_reconciliation": state.checkpoint.requires_reconciliation}),
        principal_id=state.principal_id, organization_id=state.organization_id,
    ))
    db.commit()


def accept_task(db: Session, state: AgentRunState, payload: dict[str, Any], principal: Principal) -> None:
    """Commit acceptance before the worker is scheduled, including the scoped run."""
    _scope(state, principal)
    state.checkpoint = state.checkpoint or ExecutionCheckpoint()
    state.checkpoint.data["request_payload_sha256"] = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    execution = TaskExecutionRecord(
        run_id=state.id, payload_json=json.dumps(payload), phase="pending", revision=0,
        lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=LEASE_SECONDS),
    )
    # Flush the parent first for databases that enforce the FK immediately.
    row = AgentRunRecord(id=state.id, request=state.request, status=state.status.value,
                         state_json=state.model_dump_json(), created_at=state.created_at,
                         updated_at=state.updated_at, organization_id=state.organization_id,
                         owner_id=state.principal_id, workspace_id=state.workspace_id,
                         department_id=state.department_id, classification=state.classification)
    db.add(row)
    db.flush()
    db.add(execution)
    _append(db, state, execution, "accepted", "pending")


class CheckpointWriter:
    def __init__(self, db: Session, state: AgentRunState, principal: Principal) -> None:
        self.db, self.state, self.principal = db, state, principal
        self.owner_token = str(uuid4())
        self.cancellation_event: asyncio.Event | None = None
        self.workspace_root = None
        self.authorization_check = None
        self.started = time.monotonic()
        self.base_active_seconds = float(state.runtime_metrics.get("active_seconds", 0))

    def claim(self) -> None:
        changed = self.db.execute(update(TaskExecutionRecord).where(
            TaskExecutionRecord.run_id == self.state.id,
            TaskExecutionRecord.phase == "pending", TaskExecutionRecord.owner_token.is_(None),
        ).values(phase="active", owner_token=self.owner_token,
                 lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=LEASE_SECONDS)))
        self.db.commit()
        if changed.rowcount != 1:
            raise RuntimeError("Task execution is already claimed or no longer pending")

    def claim_resume(self, revision: int) -> None:
        changed = self.db.execute(update(TaskExecutionRecord).where(
            TaskExecutionRecord.run_id == self.state.id,
            TaskExecutionRecord.phase.in_(["paused", "interrupted", "finished"]),
            TaskExecutionRecord.owner_token.is_(None), TaskExecutionRecord.revision == revision,
        ).values(phase="active", owner_token=self.owner_token, cancel_requested=False,
                 lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=LEASE_SECONDS)),
            execution_options={"synchronize_session": False})
        if changed.rowcount != 1:
            self.db.rollback()
            raise RuntimeError("RESUME_STALE_REVISION")
        # save() commits the claim together with the reconciled snapshot.

    def save(self, state: AgentRunState, boundary: str, *, final: bool = False) -> None:
        execution = self.db.get(TaskExecutionRecord, self.state.id, populate_existing=True)
        if execution is None or execution.owner_token != self.owner_token or execution.phase != "active":
            raise RuntimeError("Task execution lease is no longer owned by this worker")
        # The conditional write holds the row lock through the snapshot commit,
        # so recovery cannot fence this worker between the ownership check and save.
        changed = self.db.execute(update(TaskExecutionRecord).where(
            TaskExecutionRecord.run_id == self.state.id,
            TaskExecutionRecord.owner_token == self.owner_token,
            TaskExecutionRecord.phase == "active",
        ).values(lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=LEASE_SECONDS)),
            execution_options={"synchronize_session": False})
        if changed.rowcount != 1:
            raise RuntimeError("Task execution lease is no longer owned by this worker")
        if state is not self.state:
            state.id = self.state.id
            state.checkpoint = self.state.checkpoint
            for name in ("principal_id", "organization_id", "workspace_id", "department_id", "classification"):
                setattr(state, name, getattr(self.state, name))
        self.state = state
        _scope(state, self.principal)
        if state.status == RunStatus.completed and state.final_response:
            state.checkpoint = state.checkpoint or ExecutionCheckpoint()
            state.checkpoint.data.update(draft_response=state.final_response, draft_ready=True)
        state.runtime_metrics["active_seconds"] = self.base_active_seconds + time.monotonic() - self.started
        execution.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=LEASE_SECONDS)
        phase = "active"
        if final:
            assess_lifecycle(state, touch=True)
            phase = "paused" if state.status in PAUSED_STATUSES else "finished"
            execution.owner_token = None
            execution.lease_expires_at = None
        _append(self.db, state, execution, boundary, phase)

    async def heartbeat(self, cancellation_event: asyncio.Event) -> None:
        # Separate session: never commit tool work that has not reached a boundary.
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            with Session(self.db.get_bind()) as db:
                changed = db.execute(update(TaskExecutionRecord).where(
                    TaskExecutionRecord.run_id == self.state.id,
                    TaskExecutionRecord.owner_token == self.owner_token,
                    TaskExecutionRecord.phase == "active",
                ).values(lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=LEASE_SECONDS)))
                db.commit()
                execution = db.get(TaskExecutionRecord, self.state.id)
                if execution and execution.cancel_requested:
                    cancellation_event.set()
                    return
                if changed.rowcount != 1:
                    cancellation_event.set()
                    return

    def stop(self, status: RunStatus, reason: str) -> AgentRunState:
        self.db.rollback()
        state = self.state
        state.status = status
        state.checkpoint = state.checkpoint or ExecutionCheckpoint()
        state.checkpoint.interruption_reason = reason
        _mark_interrupted(state, cancelled=status == RunStatus.cancelled)
        # Keep tool receipts, artifacts, evidence and successful steps intact.
        if state.final_response:
            state.checkpoint.data["draft_response"] = state.final_response
        state.final_response = reason
        state.warnings.append(reason)
        state.lifecycle = None
        self.save(state, "task_cancelled" if status == RunStatus.cancelled else "task_failed", final=True)
        return state


@contextmanager
def checkpoint_context(writer: CheckpointWriter):
    token = _writer.set(writer)
    try:
        yield writer
    finally:
        _writer.reset(token)


async def checkpoint(state: AgentRunState, boundary: str) -> bool:
    writer = _writer.get()
    if writer:
        writer.save(state, boundary)
        return True
    return False


def ensure_active() -> None:
    writer = _writer.get()
    if writer:
        if writer.authorization_check:
            writer.authorization_check(writer.principal, writer.state)
        execution = writer.db.get(TaskExecutionRecord, writer.state.id, populate_existing=True)
        if execution is None or execution.phase != "active" or execution.owner_token != writer.owner_token:
            raise RuntimeError("Task execution lease is no longer owned by this worker")
        if execution.cancel_requested or (writer.cancellation_event and writer.cancellation_event.is_set()):
            raise ModelGenerationCancelled("Task execution was cancelled before the next action")


def receipt_context(state: AgentRunState, record: dict):
    from app.agent.receipts import ActionContext
    writer = _writer.get()
    if writer and writer.workspace_root:
        return ActionContext(writer.db, state.id, record["operation_id"], record["tool"],
                             record["action_hash"], writer.workspace_root)
    return None


async def checkpoint_data(boundary: str, **data: Any) -> None:
    writer = _writer.get()
    if writer:
        state = writer.state
        state.checkpoint = state.checkpoint or ExecutionCheckpoint()
        state.checkpoint.data.update(_plain(data))
        writer.save(state, boundary)


async def operation_started(key: str, arguments: dict[str, Any]) -> None:
    ensure_active()
    writer = _writer.get()
    if writer:
        writer.state.checkpoint.data.setdefault("active_operations", {})[key] = _plain(arguments)
        writer.save(writer.state, f"{key.split(':')[0]}_started")


async def operation_finished(key: str, output: Any) -> None:
    writer = _writer.get()
    if writer:
        writer.state.checkpoint.data.setdefault("active_operations", {}).pop(key, None)
        writer.state.checkpoint.data.setdefault("operation_outputs", {})[key] = _plain(output)
        writer.save(writer.state, f"{key.split(':')[0]}_result_saved")


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude={"checkpoint"} if isinstance(value, AgentRunState) else set())
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _mark_interrupted(state: AgentRunState, *, cancelled: bool = False) -> None:
    for record in state.tool_records:
        if record.get("execution_status") == "in_flight":
            record.update(execution_status="unknown", success=None,
                          error="Execution stopped before a durable tool result was recorded.")
    for step in state.plan.steps:
        if step.status == StepStatus.running:
            step.status = StepStatus.cancelled if cancelled else StepStatus.interrupted
    for entry in state.structured_plan.get("steps", []):
        if entry.get("status") == "running":
            entry["status"] = "cancelled" if cancelled else "interrupted"


def recover_expired_tasks(db: Session, run_id: str | None = None) -> int:
    """Fence abandoned workers atomically. Live leases, pauses and final runs stay intact."""
    now = datetime.now(timezone.utc)
    query = select(TaskExecutionRecord).where(
        TaskExecutionRecord.phase.in_(["pending", "active"]),
        TaskExecutionRecord.lease_expires_at <= now,
    )
    if run_id:
        query = query.where(TaskExecutionRecord.run_id == run_id)
    recovered = 0
    for execution in db.scalars(query).all():
        last_heartbeat = execution.lease_expires_at
        changed = db.execute(update(TaskExecutionRecord).where(
            TaskExecutionRecord.run_id == execution.run_id,
            TaskExecutionRecord.phase.in_(["pending", "active"]),
            TaskExecutionRecord.lease_expires_at <= now,
        ).values(phase="interrupted", owner_token=None, lease_expires_at=None),
            execution_options={"synchronize_session": False})
        if changed.rowcount != 1:
            continue
        db.refresh(execution)
        row = db.get(AgentRunRecord, execution.run_id, populate_existing=True)
        if row is None:
            continue
        state = AgentRunState.model_validate_json(row.state_json)
        state.status = RunStatus.interrupted
        state.checkpoint = state.checkpoint or ExecutionCheckpoint()
        # Count active time up to the last worker heartbeat, excluding downtime.
        if state.checkpoint.saved_at and last_heartbeat:
            heartbeat_at = last_heartbeat.replace(tzinfo=timezone.utc) - timedelta(seconds=LEASE_SECONDS)
            elapsed = max(0, (heartbeat_at - state.checkpoint.saved_at.replace(tzinfo=timezone.utc)).total_seconds())
            state.runtime_metrics["active_seconds"] = float(state.runtime_metrics.get("active_seconds", 0)) + elapsed
        state.checkpoint.interruption_reason = "Worker lease expired; saved progress is retained. Review unknown action outcomes before resuming."
        _mark_interrupted(state)
        state.lifecycle = None
        _append(db, state, execution, "worker_interrupted", "interrupted")
        recovered += 1
    return recovered


def stop_saved_task(db: Session, run_id: str, principal: Principal, status: RunStatus, reason: str,
                    *, allow_interrupted: bool = False) -> AgentRunState | None:
    """Finalize an unstarted or paused task without invoking any executor."""
    execution = db.get(TaskExecutionRecord, run_id, populate_existing=True)
    row = db.get(AgentRunRecord, run_id, populate_existing=True)
    if not row:
        return None
    state = AgentRunState.model_validate_json(row.state_json)
    if not execution or execution.phase == "finished":
        return state
    if execution.phase == "interrupted" and not allow_interrupted:
        return state
    if execution.phase == "active":
        return state  # Only its owner may write a terminal outcome for a live worker.
    changed = db.execute(update(TaskExecutionRecord).where(
        TaskExecutionRecord.run_id == run_id, TaskExecutionRecord.phase == execution.phase,
        TaskExecutionRecord.revision == execution.revision,
    ).values(phase="finished", owner_token=None, lease_expires_at=None),
        execution_options={"synchronize_session": False})
    if changed.rowcount != 1:
        db.rollback()
        return AgentRunState.model_validate_json(db.get(AgentRunRecord, run_id, populate_existing=True).state_json)
    db.refresh(execution)
    state.status = status
    state.checkpoint = state.checkpoint or ExecutionCheckpoint()
    state.checkpoint.interruption_reason = reason
    _mark_interrupted(state, cancelled=status == RunStatus.cancelled)
    state.final_response = reason
    state.lifecycle = None
    execution.owner_token = None
    execution.lease_expires_at = None
    assess_lifecycle(state)
    _append(db, state, execution, "task_cancelled" if status == RunStatus.cancelled else "task_failed", "finished")
    return state
