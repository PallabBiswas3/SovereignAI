from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.database import AuditEventRecord


class AuditLogger:
    def __init__(self, session: Session) -> None:
        self.session = session

    def log(self, run_id: str, event_type: str, summary: str, payload: dict[str, Any] | None = None) -> AuditEventRecord:
        event = AuditEventRecord(
            id=str(uuid4()), run_id=run_id, event_type=event_type,
            summary=summary[:500], payload_json=json.dumps(payload or {}, ensure_ascii=False, default=str),
        )
        self.session.add(event)
        self.session.commit()
        return event

