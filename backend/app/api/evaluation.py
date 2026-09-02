from __future__ import annotations

import json
from uuid import uuid4

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import MetricSnapshotRecord, get_db
from app.evaluation.runner import EvaluationRunner
from app.identity.dependencies import require_permission
from app.identity.models import Permission, Principal


router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.post("/run")
async def run_evaluation(
    principal: Principal = Depends(require_permission(Permission.audit_read)),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    settings = get_settings()
    metrics = EvaluationRunner(db, settings.models_config, settings.knowledge_root).run()
    snapshot = MetricSnapshotRecord(id=str(uuid4()), metrics_json=json.dumps(metrics))
    db.add(snapshot)
    db.commit()
    return {"snapshot_id": snapshot.id, "metrics": metrics}


@router.get("/metrics")
async def latest_metrics(
    principal: Principal = Depends(require_permission(Permission.audit_read)),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    snapshot = db.query(MetricSnapshotRecord).order_by(MetricSnapshotRecord.created_at.desc()).first()
    if snapshot:
        return {"snapshot_id": snapshot.id, "created_at": snapshot.created_at.isoformat(), "metrics": json.loads(snapshot.metrics_json)}
    settings = get_settings()
    return {"snapshot_id": None, "created_at": None, "metrics": EvaluationRunner(db, settings.models_config, settings.knowledge_root).run()}
