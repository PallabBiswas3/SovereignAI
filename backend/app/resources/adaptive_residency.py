from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from app.resources.footprints import ModelFootprintStore
from app.resources.hardware import LocalHardwareProbe
from app.resources.residency import OllamaResidencyManager, ResidencyAction


@dataclass(frozen=True, slots=True)
class ResidencyDecision:
    target_model: str
    target_was_resident: bool
    estimated_load_mb: float | None
    available_before_mb: float
    reserve_mb: float
    evicted_models: tuple[str, ...] = field(default_factory=tuple)
    actions: tuple[ResidencyAction, ...] = field(default_factory=tuple)
    safe_to_load: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "target_model": self.target_model,
            "target_was_resident": self.target_was_resident,
            "estimated_load_mb": self.estimated_load_mb,
            "available_before_mb": self.available_before_mb,
            "reserve_mb": self.reserve_mb,
            "evicted_models": list(self.evicted_models),
            "actions": [item.to_dict() for item in self.actions],
            "safe_to_load": self.safe_to_load,
            "reason": self.reason,
        }


class AdaptiveResidencyManager:
    """RAM-aware reconciliation for local Ollama model residency.

    The requested model is never evicted. Other resident models are considered only when the
    target is not already loaded and its measured footprint would violate the configured RAM
    reserve. This controller deliberately avoids speculative multi-model warming on laptops.
    """

    def __init__(
        self,
        residency: OllamaResidencyManager,
        footprints: ModelFootprintStore,
        *,
        hardware_probe: LocalHardwareProbe | None = None,
        ram_reserve_mb: float = 1536.0,
        ram_reserve_fraction: float = 0.15,
    ) -> None:
        self.residency = residency
        self.footprints = footprints
        self.hardware_probe = hardware_probe or LocalHardwareProbe()
        self.ram_reserve_mb = max(0.0, float(ram_reserve_mb))
        self.ram_reserve_fraction = min(0.90, max(0.0, float(ram_reserve_fraction)))

    async def reconcile_for(self, target_model: str) -> ResidencyDecision:
        hardware = self.hardware_probe.snapshot()
        reserve = max(self.ram_reserve_mb, hardware.ram_total_mb * self.ram_reserve_fraction)
        residents = await self.residency.list_resident_models()
        target_names = _normalized_names(target_model)
        target_resident = any(_resident_name(item) in target_names for item in residents)
        estimate = self.footprints.admission_estimate_mb(target_model)

        if target_resident:
            return ResidencyDecision(
                target_model=target_model,
                target_was_resident=True,
                estimated_load_mb=0.0,
                available_before_mb=hardware.ram_available_mb,
                reserve_mb=round(reserve, 2),
                safe_to_load=True,
                reason="Target model is already resident; no additional model-load budget is charged.",
            )

        if estimate is None:
            return ResidencyDecision(
                target_model=target_model,
                target_was_resident=False,
                estimated_load_mb=None,
                available_before_mb=hardware.ram_available_mb,
                reserve_mb=round(reserve, 2),
                safe_to_load=hardware.ram_available_mb >= reserve,
                reason=(
                    "No measured footprint is available yet; preserve the OS RAM reserve and "
                    "let the benchmark establish a model-specific estimate."
                ),
            )

        if hardware.ram_available_mb >= reserve + estimate:
            return ResidencyDecision(
                target_model=target_model,
                target_was_resident=False,
                estimated_load_mb=estimate,
                available_before_mb=hardware.ram_available_mb,
                reserve_mb=round(reserve, 2),
                safe_to_load=True,
                reason="Measured target footprint fits while preserving the RAM reserve.",
            )

        actions: list[ResidencyAction] = []
        evicted: list[str] = []
        candidates = [
            item for item in residents if _resident_name(item) and _resident_name(item) not in target_names
        ]
        # Prefer evicting the largest resident model first; it minimizes unload churn.
        candidates.sort(key=_resident_size_bytes, reverse=True)

        for item in candidates:
            name = _resident_name(item)
            if not name:
                continue
            action = await self.residency.unload(name)
            actions.append(action)
            if action.resident_before and not action.resident_after:
                evicted.append(name)
            hardware = self.hardware_probe.snapshot()
            if hardware.ram_available_mb >= reserve + estimate:
                return ResidencyDecision(
                    target_model=target_model,
                    target_was_resident=False,
                    estimated_load_mb=estimate,
                    available_before_mb=hardware.ram_available_mb,
                    reserve_mb=round(reserve, 2),
                    evicted_models=tuple(evicted),
                    actions=tuple(actions),
                    safe_to_load=True,
                    reason="Evicted idle resident model(s) until the measured target footprint fit.",
                )

        hardware = self.hardware_probe.snapshot()
        return ResidencyDecision(
            target_model=target_model,
            target_was_resident=False,
            estimated_load_mb=estimate,
            available_before_mb=hardware.ram_available_mb,
            reserve_mb=round(reserve, 2),
            evicted_models=tuple(evicted),
            actions=tuple(actions),
            safe_to_load=hardware.ram_available_mb >= reserve + estimate,
            reason=(
                "Insufficient RAM for the measured target footprint after evicting eligible resident models."
            ),
        )


def _resident_name(item: dict[str, object]) -> str:
    return str(item.get("name") or item.get("model") or "").strip()


def _resident_size_bytes(item: dict[str, object]) -> int:
    value = item.get("size")
    return int(value) if isinstance(value, (int, float)) and value > 0 else 0


def _normalized_names(model: str) -> set[str]:
    value = model.strip()
    if not value:
        return set()
    return {value, f"{value}:latest"} if ":" not in value else {value}
