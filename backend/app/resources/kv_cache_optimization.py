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


def select_kv_cache_result(
    results: list[dict[str, Any]],
    *,
    min_quality: float,
    max_tps_regression: float,
    min_ram_saving_mb: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose the most memory-efficient cache that stays inside quality/performance guardrails."""
    baseline = next((item for item in results if item.get("cache_type") == "f16"), None)
    if baseline is None:
        raise ValueError("f16 baseline result is required")
    base_tps = float(baseline.get("median_tokens_per_second") or 0.0)
    base_ram = float(baseline.get("system_ram_delta_mb") or 0.0)
    if base_tps <= 0:
        raise ValueError("f16 baseline requires positive median throughput")

    eligible: list[tuple[float, float, int, dict[str, Any]]] = []
    rank = {"f16": 0, "q8_0": 1, "q4_0": 2}
    for item in results:
        cache_type = str(item.get("cache_type") or "")
        if cache_type not in rank:
            raise ValueError(f"unsupported KV cache type in benchmark result: {cache_type}")
        tps = float(item.get("median_tokens_per_second") or 0.0)
        ram = float(item.get("system_ram_delta_mb") or 0.0)
        quality = float(item.get("quality_score") or 0.0)
        tps_ratio = tps / base_tps if tps > 0 else 0.0
        ram_saving = max(0.0, base_ram - ram)
        item["tps_ratio_vs_f16"] = round(tps_ratio, 4) if tps_ratio else None
        item["ram_saving_vs_f16_mb"] = round(ram_saving, 2)
        qualifies = (
            cache_type == "f16"
            or (
                quality >= min_quality
                and tps_ratio >= 1.0 - max_tps_regression
                and ram_saving >= min_ram_saving_mb
            )
        )
        item["eligible"] = qualifies
        if qualifies:
            eligible.append((ram_saving, tps_ratio, rank[cache_type], item))

    # Maximize RAM saved first. If savings are effectively tied, prefer throughput and then the
    # less aggressive representation so q8_0 wins a tie over q4_0.
    selected = max(eligible, key=lambda value: (value[0], value[1], -value[2]))[3]
    policy = {
        "minimum_quality": min_quality,
        "maximum_tps_regression": max_tps_regression,
        "minimum_ram_saving_mb": min_ram_saving_mb,
        "selection_priority": "maximize RAM saving subject to quality and throughput guardrails",
    }
    return selected, policy


def _key(value: str) -> str:
    return value.strip().lower()
