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
    # Incremental OS-visible pressure expected when loading a non-resident model.
    admission_estimate_mb: float = Field(gt=0)
    # Total resident footprint is tracked separately because mmap/file-backed pages can make
    # process RSS substantially larger than the reduction in system available RAM.
    resident_estimate_mb: float | None = Field(default=None, ge=0)
    admission_basis: str = "legacy"
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

    def resident_estimate_mb(self, model: str) -> float | None:
        profile = self.get(model)
        return profile.resident_estimate_mb if profile else None

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
        data["version"] = 2
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
            return {"version": 2, "models": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 2, "models": {}}
        if not isinstance(data, dict):
            return {"version": 2, "models": {}}
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
    """Build two distinct memory estimates from a local CPU benchmark.

    Admission should model *incremental system pressure*, not total process RSS. Ollama uses
    memory mapping, so process RSS/model size can exceed the observed reduction in OS available
    RAM. When an OS-visible delta is available we therefore use it for admission, with the
    configured safety multiplier. The largest process/Ollama size remains available separately
    as the resident footprint for reporting and eviction heuristics.
    """

    observed_system_delta = _positive(system_ram_delta_mb)
    resident_candidates = [
        value
        for value in (
            ollama_process_rss_delta_mb,
            ollama_reported_cpu_size_mb,
            ollama_reported_size_mb,
        )
        if _positive(value) is not None
    ]
    resident_estimate = max(float(value) for value in resident_candidates) if resident_candidates else None

    if observed_system_delta is not None:
        admission_base = observed_system_delta
        admission_basis = "system_ram_delta"
    elif resident_estimate is not None:
        # Fallback for platforms where a reliable system delta could not be measured.
        admission_base = resident_estimate
        admission_basis = "resident_size_fallback"
    else:
        raise ValueError("At least one positive model-memory measurement is required")

    multiplier = max(1.0, safety_multiplier)
    return ModelFootprintProfile(
        model=model,
        sample_count=sample_count,
        system_ram_delta_mb=_round_optional(system_ram_delta_mb),
        ollama_process_rss_delta_mb=_round_optional(ollama_process_rss_delta_mb),
        ollama_reported_size_mb=_round_optional(ollama_reported_size_mb),
        ollama_reported_cpu_size_mb=_round_optional(ollama_reported_cpu_size_mb),
        context_length=context_length,
        admission_estimate_mb=round(admission_base * multiplier, 2),
        resident_estimate_mb=round(resident_estimate, 2) if resident_estimate is not None else None,
        admission_basis=admission_basis,
        safety_multiplier=multiplier,
    )


def _positive(value: float | None) -> float | None:
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def _round_optional(value: float | None) -> float | None:
    return round(float(value), 2) if isinstance(value, (int, float)) else None


def _model_key(model: str) -> str:
    return model.strip().lower()
