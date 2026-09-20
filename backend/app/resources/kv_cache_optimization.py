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


class KvCacheContextConfirmation(BaseModel):
    model: str
    context_length: int = Field(gt=0)
    baseline_cache: KvCacheType = "f16"
    candidate_cache: KvCacheType = "q8_0"
    confirmed: bool
    paired_speed_ratio: float | None = Field(default=None, gt=0)
    candidate_win_rate: float = Field(ge=0.0, le=1.0)
    resident_saving_mb: float | None = None
    baseline_quality_score: float = Field(ge=0.0, le=1.0)
    candidate_quality_score: float = Field(ge=0.0, le=1.0)
    long_context_passed: bool
    reasons: list[str] = Field(default_factory=list)
    measured_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "controlled-kv-cache-confirmation"


class KvCacheOptimizationStore:
    """Persist the target-machine global KV-cache choice without mutating system environment variables."""

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
        _atomic_write(self.path, data)

    def _read(self) -> dict[str, Any]:
        return _read_json_store(self.path, root_key="models")


class KvCacheConfirmationStore:
    """Persist per-context confirmation evidence separately from the active global server profile."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self, model: str, context_length: int) -> KvCacheContextConfirmation | None:
        model_entry = self._read().get("models", {}).get(_key(model))
        if not isinstance(model_entry, dict):
            return None
        raw = model_entry.get("contexts", {}).get(str(context_length))
        if not isinstance(raw, dict):
            return None
        try:
            return KvCacheContextConfirmation.model_validate(raw)
        except Exception:
            return None

    def save(self, confirmation: KvCacheContextConfirmation) -> None:
        data = self._read()
        model_entry = data.setdefault("models", {}).setdefault(_key(confirmation.model), {})
        model_entry.setdefault("contexts", {})[str(confirmation.context_length)] = confirmation.model_dump(mode="json")
        data["version"] = 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.path, data)

    def _read(self) -> dict[str, Any]:
        return _read_json_store(self.path, root_key="models")


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

    selected = max(eligible, key=lambda value: (value[0], value[1], -value[2]))[3]
    policy = {
        "minimum_quality": min_quality,
        "maximum_tps_regression": max_tps_regression,
        "minimum_ram_saving_mb": min_ram_saving_mb,
        "selection_priority": "maximize RAM saving subject to quality and throughput guardrails",
    }
    return selected, policy


def evaluate_context_confirmation(
    *,
    model: str,
    context_length: int,
    paired_speed_ratios: list[float],
    baseline_resident_sizes_mb: list[float],
    candidate_resident_sizes_mb: list[float],
    baseline_quality_score: float,
    candidate_quality_score: float,
    baseline_long_context_passed: bool,
    candidate_long_context_passed: bool,
    min_quality: float = 0.95,
    min_speed_ratio: float = 1.03,
    min_win_rate: float = 0.75,
    min_resident_saving_mb: float = 128.0,
) -> KvCacheContextConfirmation:
    """Validate q8_0 for one context using paired speed and resident-size evidence.

    OS available-RAM deltas are intentionally excluded from the decision because Windows background
    memory pressure made the Phase 10 v1 baselines non-comparable. They remain diagnostic telemetry.
    """
    from statistics import median

    valid_ratios = [float(value) for value in paired_speed_ratios if value > 0]
    baseline_sizes = [float(value) for value in baseline_resident_sizes_mb if value > 0]
    candidate_sizes = [float(value) for value in candidate_resident_sizes_mb if value > 0]
    paired_ratio = median(valid_ratios) if valid_ratios else None
    win_rate = (sum(1 for value in valid_ratios if value > 1.0) / len(valid_ratios)) if valid_ratios else 0.0
    resident_saving = (
        median(baseline_sizes) - median(candidate_sizes)
        if baseline_sizes and candidate_sizes
        else None
    )

    reasons: list[str] = []
    if baseline_quality_score < min_quality:
        reasons.append(f"f16 quality {baseline_quality_score:.3f} is below {min_quality:.3f}")
    if candidate_quality_score < min_quality:
        reasons.append(f"q8_0 quality {candidate_quality_score:.3f} is below {min_quality:.3f}")
    if paired_ratio is None or paired_ratio < min_speed_ratio:
        reasons.append(
            f"paired speed ratio {paired_ratio if paired_ratio is not None else 'missing'} is below {min_speed_ratio:.3f}"
        )
    if win_rate < min_win_rate:
        reasons.append(f"q8_0 win rate {win_rate:.3f} is below {min_win_rate:.3f}")
    if resident_saving is None or resident_saving < min_resident_saving_mb:
        reasons.append(
            f"resident saving {resident_saving if resident_saving is not None else 'missing'} MB is below {min_resident_saving_mb:.1f} MB"
        )
    long_context_passed = baseline_long_context_passed and candidate_long_context_passed
    if not long_context_passed:
        reasons.append("long-context validation did not pass on both cache types")

    return KvCacheContextConfirmation(
        model=model,
        context_length=context_length,
        confirmed=not reasons,
        paired_speed_ratio=round(paired_ratio, 4) if paired_ratio is not None else None,
        candidate_win_rate=round(win_rate, 4),
        resident_saving_mb=round(resident_saving, 2) if resident_saving is not None else None,
        baseline_quality_score=baseline_quality_score,
        candidate_quality_score=candidate_quality_score,
        long_context_passed=long_context_passed,
        reasons=reasons,
    )


def _read_json_store(path: Path, *, root_key: str) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, root_key: {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, root_key: {}}
    if not isinstance(data, dict):
        return {"version": 1, root_key: {}}
    if not isinstance(data.get(root_key), dict):
        data[root_key] = {}
    return data


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    with NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp"
    ) as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _key(value: str) -> str:
    return value.strip().lower()
