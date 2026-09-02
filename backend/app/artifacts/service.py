from __future__ import annotations

from pathlib import Path
import hashlib
import json
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.database import ArtifactRecord
from app.identity.models import ResourceScope


MEDIA_TYPES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".py": "text/x-python",
    ".csv": "text/csv",
    ".md": "text/markdown",
}


class ArtifactService:
    def __init__(self, session: Session, root: Path) -> None:
        self.session = session
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        path: Path,
        run_id: str | None = None,
        *,
        workcell_id: str | None = None,
        workcell_version: str | None = None,
        artifact_type: str | None = None,
        derived_from_claims: list[str] | None = None,
        scope: ResourceScope | None = None,
    ) -> ArtifactRecord:
        resolved = path.resolve()
        if self.root not in resolved.parents:
            raise ValueError("Artifact is outside the configured artifact directory")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        record = ArtifactRecord(
            id=str(uuid4()), run_id=run_id, name=resolved.name,
            media_type=MEDIA_TYPES.get(resolved.suffix.lower(), "application/octet-stream"),
            path=str(resolved.relative_to(self.root)).replace("\\", "/"), size=resolved.stat().st_size,
            sha256=digest, workcell_id=workcell_id, workcell_version=workcell_version,
            artifact_type=artifact_type or resolved.suffix.lower().lstrip("."),
            lineage_json=json.dumps({"derived_from_claims": derived_from_claims or []}),
            organization_id=scope.organization_id if scope else None,
            owner_id=scope.owner_id if scope else None,
            workspace_id=scope.workspace_id if scope else None,
            department_id=scope.department_id if scope else None,
            classification=scope.classification.name.upper() if scope else "INTERNAL",
            allowed_roles_json=json.dumps([role.value for role in scope.allowed_roles] if scope else []),
            allowed_users_json=json.dumps(scope.allowed_users if scope else []),
        )
        self.session.add(record)
        self.session.commit()
        return record

    def resolve(self, record: ArtifactRecord) -> Path:
        candidate = (self.root / record.path).resolve()
        if self.root not in candidate.parents:
            raise ValueError("Invalid artifact path")
        return candidate
