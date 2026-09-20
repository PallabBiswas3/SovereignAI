import asyncio

from app.resources.adaptive_residency import AdaptiveResidencyManager
from app.resources.footprints import ModelFootprintStore, build_measured_profile
from app.resources.hardware import HardwareSnapshot
from app.resources.residency import ResidencyAction
from app.resources.scheduler import ModelJob, ResourceScheduler

MIB = 1024 * 1024


class FakeResidency:
    def __init__(self, models: list[dict[str, object]]):
        self.models = [dict(item) for item in models]
        self.evicted: list[str] = []

    async def list_resident_models(self) -> list[dict[str, object]]:
        return [dict(item) for item in self.models]

    async def unload(self, model: str) -> ResidencyAction:
        before = any(str(item.get("name")) == model for item in self.models)
        self.models = [item for item in self.models if str(item.get("name")) != model]
        if before:
            self.evicted.append(model)
        return ResidencyAction(model, "unload", before, False, 0)


class ResidencyAwareProbe:
    def __init__(self, residency: FakeResidency, *, base_available_mb: float, total_mb: float = 16000):
        self.residency = residency
        self.base_available_mb = base_available_mb
        self.total_mb = total_mb
        self.original_bytes = sum(int(item.get("size") or 0) for item in residency.models)

    def snapshot(self) -> HardwareSnapshot:
        current_bytes = sum(int(item.get("size") or 0) for item in self.residency.models)
        freed_mb = (self.original_bytes - current_bytes) / MIB
        available = self.base_available_mb + freed_mb
        return HardwareSnapshot(
            cpu_percent=42.0,
            physical_cores=8,
            logical_cores=16,
            ram_used_mb=self.total_mb - available,
            ram_available_mb=available,
            ram_total_mb=self.total_mb,
            ram_percent=round((self.total_mb - available) / self.total_mb * 100, 2),
            swap_used_mb=0,
            swap_total_mb=0,
            vram_used_mb=None,
            vram_total_mb=None,
            vram_source=None,
        )


def test_measured_footprint_store_round_trip(tmp_path) -> None:
    store = ModelFootprintStore(tmp_path / "footprints.json")
    profile = build_measured_profile(
        "qwen3:4b-instruct",
        system_ram_delta_mb=3100,
        ollama_process_rss_delta_mb=2800,
        ollama_reported_size_mb=3300,
        ollama_reported_cpu_size_mb=3300,
        context_length=4096,
        safety_multiplier=1.10,
    )
    store.save(profile)
    loaded = store.get("QWEN3:4B-INSTRUCT")
    assert loaded is not None
    assert loaded.admission_estimate_mb == 3410.0
    assert loaded.resident_estimate_mb == 3300.0
    assert loaded.admission_basis == "system_ram_delta"
    assert store.admission_estimate_mb("qwen3:4b-instruct") == 3410.0


def test_resident_size_is_fallback_when_system_delta_is_unavailable(tmp_path) -> None:
    store = ModelFootprintStore(tmp_path / "footprints.json")
    profile = build_measured_profile(
        "qwen3:4b-instruct",
        system_ram_delta_mb=None,
        ollama_process_rss_delta_mb=2800,
        ollama_reported_size_mb=3300,
        ollama_reported_cpu_size_mb=3300,
        context_length=4096,
        safety_multiplier=1.10,
    )
    store.save(profile)
    assert profile.admission_estimate_mb == 3630.0
    assert profile.resident_estimate_mb == 3300.0
    assert profile.admission_basis == "resident_size_fallback"


def test_resident_target_is_not_double_charged(tmp_path) -> None:
    store = ModelFootprintStore(tmp_path / "footprints.json")
    store.save(build_measured_profile(
        "qwen3:4b-instruct",
        system_ram_delta_mb=4000,
        ollama_process_rss_delta_mb=None,
        ollama_reported_size_mb=None,
        ollama_reported_cpu_size_mb=None,
        context_length=4096,
    ))
    residency = FakeResidency([{"name": "qwen3:4b-instruct", "size": 4000 * MIB}])
    manager = AdaptiveResidencyManager(
        residency,
        store,
        hardware_probe=ResidencyAwareProbe(residency, base_available_mb=1800),
        ram_reserve_mb=1500,
        ram_reserve_fraction=0.05,
    )
    decision = asyncio.run(manager.reconcile_for("qwen3:4b-instruct"))
    assert decision.target_was_resident is True
    assert decision.estimated_load_mb == 0.0
    assert decision.safe_to_load is True
    assert residency.evicted == []


def test_adaptive_residency_evicts_largest_other_model_until_target_fits(tmp_path) -> None:
    store = ModelFootprintStore(tmp_path / "footprints.json")
    store.save(build_measured_profile(
        "qwen3:4b-instruct",
        system_ram_delta_mb=3600,
        ollama_process_rss_delta_mb=None,
        ollama_reported_size_mb=None,
        ollama_reported_cpu_size_mb=None,
        context_length=4096,
        safety_multiplier=1.0,
    ))
    residency = FakeResidency([
        {"name": "small:latest", "size": 1000 * MIB},
        {"name": "large:latest", "size": 3000 * MIB},
    ])
    probe = ResidencyAwareProbe(residency, base_available_mb=2500)
    manager = AdaptiveResidencyManager(
        residency,
        store,
        hardware_probe=probe,
        ram_reserve_mb=1500,
        ram_reserve_fraction=0.05,
    )
    decision = asyncio.run(manager.reconcile_for("qwen3:4b-instruct"))
    assert decision.safe_to_load is True
    assert decision.evicted_models == ("large:latest",)
    assert residency.evicted == ["large:latest"]


def test_scheduler_charges_zero_incremental_model_ram_when_resident() -> None:
    class Probe:
        def snapshot(self) -> HardwareSnapshot:
            return HardwareSnapshot(
                cpu_percent=25,
                physical_cores=4,
                logical_cores=8,
                ram_used_mb=7000,
                ram_available_mb=1800,
                ram_total_mb=8800,
                ram_percent=79.5,
                swap_used_mb=0,
                swap_total_mb=0,
                vram_used_mb=None,
                vram_total_mb=None,
                vram_source=None,
            )

    async def run():
        scheduler = ResourceScheduler(
            max_model_jobs=1,
            ram_reserve_mb=1500,
            ram_reserve_fraction=0.10,
            hardware_probe=Probe(),
        )
        async with scheduler.acquire_model(
            ModelJob(model="qwen3:4b-instruct", resident=True, estimated_ram_mb=4000)
        ) as permit:
            return permit

    permit = asyncio.run(run())
    assert permit.admission_mode == "resident-no-load"
    assert permit.estimated_model_ram_mb == 0.0
