from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from pydantic import BaseModel, Field


KvCacheType = Literal["f16", "q8_0", "q4_0"]


class KvCacheOptimizationProfile(BaseModel):
    model: str
    cache_type: KvCacheType
    flash_attention: bool = True
    context_length: int = Field(gt=0)
    quality_score: float = Field(ge=0.0, le=1.0)
    median_tokens_per_second: float | None = Field(default=None, gt=0)
    resident_size_mb: float | None = Field(default=None, gt=0)
    system_ram_delta_mb: float | None = Field(default=None, ge=0)
    available_ram_after_mb: float | None = Field(default=None, ge=0)
    measured_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "local-kv-cache-benchmark"


class KvCacheOptimizationStore:
    """Persist the target-machine KV-cache choice without mutating system environment variables."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self, model: str) -> KvCacheOptimizationProfile | None:
        raw = self._read().get("models", {}).get(_key(model))
        if not isinstance(raw, dict):
            return None
        try:
            return KvCacheOptimizationProfile.model_validate(raw)
        except Exception:
            return None

    def save(self, profile: KvCacheOptimizationProfile) -> None:
        data = self._read()
        data.setdefault("models", {})[_key(profile.model)] = profile.model_dump(mode="json")
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
