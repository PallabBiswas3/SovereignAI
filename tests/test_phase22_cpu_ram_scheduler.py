import asyncio

from app.resources.hardware import HardwareSnapshot
from app.resources.scheduler import ModelJob, ResourceAdmissionError, ResourceScheduler


class FakeHardwareProbe:
    def __init__(self, *, available_mb: float, total_mb: float = 16000.0, cpu_percent: float = 35.0):
        self.available_mb = available_mb
        self.total_mb = total_mb
        self.cpu_percent = cpu_percent

    def snapshot(self) -> HardwareSnapshot:
        return HardwareSnapshot(
            cpu_percent=self.cpu_percent,
            physical_cores=8,
            logical_cores=16,
            ram_used_mb=self.total_mb - self.available_mb,
            ram_available_mb=self.available_mb,
            ram_total_mb=self.total_mb,
            ram_percent=round((self.total_mb - self.available_mb) / self.total_mb * 100.0, 2),
            swap_used_mb=0.0,
            swap_total_mb=0.0,
            vram_used_mb=None,
            vram_total_mb=None,
            vram_source=None,
        )


def test_scheduler_admits_cpu_model_with_safe_ram_headroom() -> None:
    async def run():
        scheduler = ResourceScheduler(
            max_model_jobs=1,
            ram_reserve_mb=1500,
            ram_reserve_fraction=0.10,
            hardware_probe=FakeHardwareProbe(available_mb=8000),
        )
        async with scheduler.acquire_model(
            ModelJob(model="qwen", estimated_ram_mb=3500)
        ) as permit:
            snapshot = scheduler.snapshot()
            return permit, snapshot

    permit, snapshot = asyncio.run(run())
    assert permit.admission_mode == "ram-aware"
    assert permit.estimated_model_ram_mb == 3500
    assert permit.ram_available_mb == 8000
    assert snapshot.active_model_jobs == 1
    assert snapshot.active_gpu_jobs == 1  # compatibility alias
    assert snapshot.vram_total_mb is None
    assert snapshot.physical_cores == 8
    assert snapshot.logical_cores == 16


def test_scheduler_rejects_job_that_would_consume_ram_reserve() -> None:
    async def run():
        scheduler = ResourceScheduler(
            max_model_jobs=1,
            ram_reserve_mb=1500,
            ram_reserve_fraction=0.10,
            hardware_probe=FakeHardwareProbe(available_mb=4200),
        )
        async with scheduler.acquire_model(
            ModelJob(model="qwen", estimated_ram_mb=3500)
        ):
            raise AssertionError("Job should not have been admitted")

    try:
        asyncio.run(run())
    except ResourceAdmissionError as exc:
        assert "4200 MB RAM available" in str(exc)
        assert "3500 MB model estimate" in str(exc)
        return
    raise AssertionError("Expected ResourceAdmissionError")


def test_unknown_model_memory_keeps_serialized_safe_mode() -> None:
    async def run():
        scheduler = ResourceScheduler(
            max_model_jobs=1,
            ram_reserve_mb=1500,
            ram_reserve_fraction=0.10,
            hardware_probe=FakeHardwareProbe(available_mb=5000),
        )
        async with scheduler.acquire_model(ModelJob(model="unknown")) as permit:
            return permit

    permit = asyncio.run(run())
    assert permit.admission_mode == "serialized-unknown-model-ram"
    assert permit.estimated_model_ram_mb is None


def test_fractional_reserve_uses_larger_of_absolute_or_fraction() -> None:
    scheduler = ResourceScheduler(
        ram_reserve_mb=1000,
        ram_reserve_fraction=0.20,
        hardware_probe=FakeHardwareProbe(available_mb=10000, total_mb=32000),
    )
    snapshot = scheduler.snapshot()
    assert snapshot.ram_reserve_mb == 6400
