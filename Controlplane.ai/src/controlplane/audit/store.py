from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from uuid import uuid4

from controlplane.actions.response_transformer import redact_response
from controlplane.policy.models import PolicyProfile
from controlplane.schema import AuditEvent, CheckReport, FeedbackEvent


class JsonlAuditStore:
    def __init__(
        self,
        path: str | Path = "results/controlplane/audit.jsonl",
        *,
        store_raw_sensitive_payloads: bool = False,
    ) -> None:
        self.path = Path(path)
        self.store_raw_sensitive_payloads = store_raw_sensitive_payloads
        self._lock = Lock()

    def write(self, report: CheckReport, profile: PolicyProfile) -> str:
        event_id = f"audit-{uuid4()}"
        report.audit_event_id = event_id
        persisted_report = report.model_copy(deep=True)
        if not self.store_raw_sensitive_payloads:
            persisted_report.original_response = redact_response(
                persisted_report.original_response,
                persisted_report.findings,
            )
        event = AuditEvent(
            id=event_id,
            policy_id=profile.id,
            policy_version=profile.version,
            report=persisted_report,
            policy_snapshot=profile.model_dump(mode="json"),
            raw_sensitive_payloads_stored=self.store_raw_sensitive_payloads,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return event_id


class JsonlFeedbackStore:
    def __init__(self, path: str | Path = "results/controlplane/feedback.jsonl") -> None:
        self.path = Path(path)
        self._lock = Lock()

    def write(self, feedback: FeedbackEvent) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(feedback.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return feedback.id
