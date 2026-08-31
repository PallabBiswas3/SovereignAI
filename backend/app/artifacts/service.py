from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.database import ArtifactRecord


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

    def register(self, path: Path, run_id: str | None = None) -> ArtifactRecord:
        resolved = path.resolve()
        if self.root not in resolved.parents:
            raise ValueError("Artifact is outside the configured artifact directory")
        record = ArtifactRecord(
            id=str(uuid4()), run_id=run_id, name=resolved.name,
            media_type=MEDIA_TYPES.get(resolved.suffix.lower(), "application/octet-stream"),
            path=str(resolved.relative_to(self.root)).replace("\\", "/"), size=resolved.stat().st_size,
        )
        self.session.add(record)
        self.session.commit()
        return record

    def resolve(self, record: ArtifactRecord) -> Path:
        candidate = (self.root / record.path).resolve()
        if self.root not in candidate.parents:
            raise ValueError("Invalid artifact path")
        return candidate

