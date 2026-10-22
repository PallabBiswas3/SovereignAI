"""Action-bound receipts; filesystem effects and database commits are distinct."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256, file_digest
import json
from pathlib import Path

from sqlalchemy.orm import Session, object_session

from app.core.database import ArtifactRecord, ToolReceiptRecord
from app.tools.base import ToolResult


@dataclass
class ActionContext:
    db: Session
    run_id: str
    operation_id: str
    tool: str
    action_hash: str
    workspace_root: Path


_action: ContextVar[ActionContext | None] = ContextVar("tool_action_receipt", default=None)


@contextmanager
def action_context(context: ActionContext | None):
    token = _action.set(context)
    try:
        yield
    finally:
        _action.reset(token)


def file_proof(root: Path, path: str) -> dict:
    base = root.resolve()
    target = (base / path).resolve()
    if base not in target.parents or not target.is_file():
        raise ValueError("Receipt file is unavailable or outside its allowed root")
    with target.open("rb") as handle:
        digest = file_digest(handle, "sha256").hexdigest()
    return {"path": target.relative_to(base).as_posix(), "sha256": digest, "size": target.stat().st_size}


def persist_receipt(result: ToolResult, *, artifact: ArtifactRecord | None = None, commit: bool = True) -> None:
    context = _action.get()
    if context is None or not result.success:
        return
    existing = context.db.get(ToolReceiptRecord, context.operation_id)
    if existing:
        if existing.run_id != context.run_id or existing.tool != context.tool or existing.action_hash != context.action_hash or existing.result_json != result.model_dump_json():
            raise ValueError("Operation already has a different execution receipt")
        return
    proofs = []
    if artifact:
        proofs = [{"kind": "artifact", "artifact_id": artifact.id, "path": artifact.path,
                   "sha256": artifact.sha256, "size": artifact.size}]
    else:
        for path in result.generated_files:
            proofs.append({"kind": "workspace", **file_proof(context.workspace_root, path)})
    raw = result.model_dump_json()
    context.db.add(ToolReceiptRecord(
        operation_id=context.operation_id, run_id=context.run_id, tool=context.tool,
        action_hash=context.action_hash, result_json=raw, result_sha256=sha256(raw.encode()).hexdigest(),
        files_json=json.dumps(proofs),
    ))
    if commit:
        context.db.commit()


def artifact_receipt(record: ArtifactRecord) -> None:
    context = _action.get()
    if context and context.tool in {"generate_docx", "generate_xlsx", "generate_pptx"}:
        if object_session(record) is not context.db:
            raise ValueError("Artifact and receipt must commit in the same database session")
        if record.run_id != context.run_id:
            raise ValueError("Artifact receipt task identity does not match")
        persist_receipt(ToolResult(success=True, output={"artifact": {
            "id": record.id, "name": record.name, "url": f"/api/artifacts/{record.id}",
        }}, generated_files=[record.path]), artifact=record, commit=False)
