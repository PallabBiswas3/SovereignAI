from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from pydantic import BaseModel, Field


BackendName = Literal["ollama", "llama_cpp"]


class BackendSelectionProfile(BaseModel):
    model: str
    backend: BackendName
    endpoint: str
    backend_model: str | None = None
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    median_tokens_per_second: float | None = Field(default=None, gt=0)
    median_wall_seconds: float | None = Field(default=None, gt=0)
    speedup_vs_ollama: float | None = Field(default=None, gt=0)
    measured_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "local-backend-benchmark"


class BackendSelectionStore:
    """Machine-local backend choice produced by the target-device benchmark."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self, model: str) -> BackendSelectionProfile | None:
        raw = self._read().get("models", {}).get(_key(model))
        if not isinstance(raw, dict):
            return None
        try:
            return BackendSelectionProfile.model_validate(raw)
        except Exception:
            return None

    def save(self, profile: BackendSelectionProfile) -> None:
        data = self._read()
        data.setdefault("models", {})[_key(profile.model)] = profile.model_dump(mode="json")
        data["version"] = 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=self.path.parent, suffix=".tmp"
        ) as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, self.path)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "models": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "models": {}}
        if not isinstance(data, dict):
            return {"version": 1, "models": {}}
        if not isinstance(data.get("models"), dict):
            data["models"] = {}
        return data


def _key(value: str) -> str:
    return value.strip().lower()
