from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pydantic import BaseModel, Field


class CpuTuningProfile(BaseModel):
    model: str
    measured_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    num_thread: int = Field(gt=0)
    num_batch: int = Field(gt=0)
    num_ctx: int | None = Field(default=None, gt=0)
    median_tokens_per_second: float | None = Field(default=None, gt=0)
    mean_tokens_per_second: float | None = Field(default=None, gt=0)
    measured_runs: int = Field(default=1, ge=1)
    warmup_runs: int = Field(default=1, ge=0)
    source: str = "local-cpu-tuning"

    def ollama_options(self) -> dict[str, int]:
        return {"num_thread": self.num_thread, "num_batch": self.num_batch}


class CpuTuningStore:
    """Machine-local CPU tuning registry.

    The backing path lives under ``backend/data`` by default, which is intentionally git-ignored.
    This lets each machine retain its own empirically best Ollama runner settings without making
    laptop-specific thread counts repository defaults.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self, model: str) -> CpuTuningProfile | None:
        raw = self._read().get("models", {}).get(_model_key(model))
        if not isinstance(raw, dict):
            return None
        try:
            return CpuTuningProfile.model_validate(raw)
        except Exception:
            return None

    def options_for(self, model: str) -> dict[str, int]:
        profile = self.get(model)
        return profile.ollama_options() if profile else {}

    def all(self) -> dict[str, CpuTuningProfile]:
        output: dict[str, CpuTuningProfile] = {}
        for key, raw in self._read().get("models", {}).items():
            if not isinstance(raw, dict):
                continue
            try:
                output[key] = CpuTuningProfile.model_validate(raw)
            except Exception:
                continue
        return output

    def save(self, profile: CpuTuningProfile) -> None:
        data = self._read()
        data.setdefault("models", {})[_model_key(profile.model)] = profile.model_dump(mode="json")
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


def _model_key(model: str) -> str:
    return model.strip().lower()
