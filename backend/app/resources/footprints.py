from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pydantic import BaseModel, Field


class ModelFootprintProfile(BaseModel):
    model: str
    measured_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sample_count: int = Field(default=1, ge=1)
    system_ram_delta_mb: float | None = Field(default=None, ge=0)
    ollama_process_rss_delta_mb: float | None = Field(default=None, ge=0)
    ollama_reported_size_mb: float | None = Field(default=None, ge=0)
    ollama_reported_cpu_size_mb: float | None = Field(default=None, ge=0)
    context_length: int | None = Field(default=None, ge=0)
    admission_estimate_mb: float = Field(gt=0)
    safety_multiplier: float = Field(default=1.10, ge=1.0)
    source: str = "local-cpu-benchmark"


class ModelFootprintStore:
    """Persistent, local-only registry of empirically measured model RAM footprints."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self, model: str) -> ModelFootprintProfile | None:
        data = self._read()
        raw = data.get("models", {}).get(_model_key(model))
        if not isinstance(raw, dict):
            return None
        try:
            return ModelFootprintProfile.model_validate(raw)
        except Exception:
            return None

    def admission_estimate_mb(self, model: str) -> float | None:
        profile = self.get(model)
        return profile.admission_estimate_mb if profile else None

    def all(self) -> dict[str, ModelFootprintProfile]:
        output: dict[str, ModelFootprintProfile] = {}
        for key, raw in self._read().get("models", {}).items():
            if not isinstance(raw, dict):
                continue
            try:
                output[key] = ModelFootprintProfile.model_validate(raw)
            except Exception:
                continue
        return output

    def save(self, profile: ModelFootprintProfile) -> None:
        data = self._read()
        models = data.setdefault("models", {})
        models[_model_key(profile.model)] = profile.model_dump(mode="json")
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


def build_measured_profile(
    model: str,
    *,
    system_ram_delta_mb: float | None,
    ollama_process_rss_delta_mb: float | None,
    ollama_reported_size_mb: float | None,
    ollama_reported_cpu_size_mb: float | None,
    context_length: int | None,
    sample_count: int = 1,
    safety_multiplier: float = 1.10,
) -> ModelFootprintProfile:
    """Build a conservative admission estimate from observed CPU/RAM measurements.

    The largest available signal is used because memory-mapped model pages, OS cache behavior,
    and process accounting can make any single measurement under-report real pressure.
    """

    candidates = [
        value
        for value in (
            system_ram_delta_mb,
            ollama_process_rss_delta_mb,
            ollama_reported_cpu_size_mb,
            ollama_reported_size_mb,
        )
        if isinstance(value, (int, float)) and value > 0
    ]
    if not candidates:
        raise ValueError("At least one positive model-memory measurement is required")
    estimate = max(candidates) * max(1.0, safety_multiplier)
    return ModelFootprintProfile(
        model=model,
        sample_count=sample_count,
        system_ram_delta_mb=_round_optional(system_ram_delta_mb),
        ollama_process_rss_delta_mb=_round_optional(ollama_process_rss_delta_mb),
        ollama_reported_size_mb=_round_optional(ollama_reported_size_mb),
        ollama_reported_cpu_size_mb=_round_optional(ollama_reported_cpu_size_mb),
        context_length=context_length,
        admission_estimate_mb=round(estimate, 2),
        safety_multiplier=max(1.0, safety_multiplier),
    )


def _round_optional(value: float | None) -> float | None:
    return round(float(value), 2) if isinstance(value, (int, float)) else None


def _model_key(model: str) -> str:
    return model.strip().lower()
