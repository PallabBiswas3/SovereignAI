from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pydantic import BaseModel, Field


class ModelOptimizationProfile(BaseModel):
    base_model: str
    selected_model: str
    quantization: str | None = None
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    tokens_per_second: float | None = Field(default=None, gt=0)
    resident_size_mb: float | None = Field(default=None, gt=0)
    context_length: int | None = Field(default=None, gt=0)
    measured_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "local-model-optimization"


class ModelOptimizationStore:
    """Machine-local selected-model registry for empirically validated quantization choices."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self, base_model: str) -> ModelOptimizationProfile | None:
        raw = self._read().get("models", {}).get(_key(base_model))
        if not isinstance(raw, dict):
            return None
        try:
            return ModelOptimizationProfile.model_validate(raw)
        except Exception:
            return None

    def selected_model(self, base_model: str) -> str:
        profile = self.get(base_model)
        return profile.selected_model if profile else base_model

    def save(self, profile: ModelOptimizationProfile) -> None:
        data = self._read()
        data.setdefault("models", {})[_key(profile.base_model)] = profile.model_dump(mode="json")
        data["version"] = 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=self.path.parent, suffix=".tmp"
        ) as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temp_path = Path(handle.name)
        os.replace(temp_path, self.path)

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
