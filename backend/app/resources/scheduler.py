from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from functools import lru_cache
from threading import BoundedSemaphore, Lock
from time import monotonic
from typing import AsyncIterator, Iterator

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.resources.hardware import HardwareSnapshot, LocalHardwareProbe


class ResourceAdmissionError(RuntimeError):
    """Raised when starting another local model job would violate RAM safety headroom."""


class ModelJob(BaseModel):
    model: str
    role: str = "GENERAL"
    memory_requirement: str = "medium"
    estimated_ram_mb: float | None = Field(default=None, gt=0)
    execution_mode: str = "STANDARD"
    priority: int = Field(default=50, ge=0, le=100)


class ResourcePermit(BaseModel):
    queue_wait_seconds: float
    queue_depth_at_admission: int
    admission_mode: str = "serialized"
    ram_available_mb: float | None = None
    ram_reserve_mb: float | None = None
    estimated_model_ram_mb: float | None = None
    cpu_percent: float | None = None
    physical_cores: int | None = None
    logical_cores: int | None = None


class ResourceSnapshot(BaseModel):
    max_model_jobs: int
    max_cpu_jobs: int
    active_model_jobs: int
    active_cpu_jobs: int
    queue_depth: int
    current_model: str | None
    last_model: str | None
    cpu_percent: float
    physical_cores: int | None
    logical_cores: int | None
    ram_used_mb: float
    ram_available_mb: float
    ram_total_mb: float
    ram_percent: float
    swap_used_mb: float
    swap_total_mb: float
    ram_reserve_mb: float
    ram_admission_enabled: bool
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    vram_source: str | None = None
    # Compatibility aliases for existing clients while the runtime becomes CPU-first.
    max_gpu_model_jobs: int
    active_gpu_jobs: int


@dataclass(slots=True)
class _Waiter:
    priority: int
    sequence: int
    enqueued_at: float
    job: ModelJob
    future: asyncio.Future[ResourcePermit]


class ResourceScheduler:
    """Process-local admission controller optimized for CPU/RAM local inference.

    Generative-model work remains serialized by default, which avoids competing CPU-bound
    generations on laptop hardware. Before admitting a model job, the scheduler protects a
    configurable RAM reserve. If an exact model RAM estimate is supplied, that estimate is also
    included in the admission requirement. Unknown model RAM is never guessed.
    """

    def __init__(
        self,
        max_model_jobs: int = 1,
        max_cpu_jobs: int = 2,
        *,
        ram_admission_enabled: bool = True,
        ram_reserve_mb: float = 1536.0,
        ram_reserve_fraction: float = 0.15,
        hardware_probe: LocalHardwareProbe | None = None,
        # Backward-compatible keyword used by earlier tests/configuration.
        max_gpu_model_jobs: int | None = None,
    ) -> None:
        if max_gpu_model_jobs is not None:
            max_model_jobs = max_gpu_model_jobs
        self.max_model_jobs = max(1, max_model_jobs)
        self.max_cpu_jobs = max(1, max_cpu_jobs)
        self.ram_admission_enabled = ram_admission_enabled
        self.ram_reserve_mb = max(0.0, float(ram_reserve_mb))
        self.ram_reserve_fraction = min(0.90, max(0.0, float(ram_reserve_fraction)))
        self.hardware_probe = hardware_probe or LocalHardwareProbe()
        self._active_model = 0
        self._active_cpu = 0
        self._waiters: list[_Waiter] = []
        self._sequence = 0
        self._current_model: str | None = None
        self._last_model: str | None = None
        self._cpu_semaphore = asyncio.Semaphore(self.max_cpu_jobs)
        self._cpu_sync_semaphore = BoundedSemaphore(self.max_cpu_jobs)
        self._cpu_state_lock = Lock()

    @property
    def queue_depth(self) -> int:
        return len(self._waiters)

    def _effective_ram_reserve(self, hardware: HardwareSnapshot) -> float:
        return max(self.ram_reserve_mb, hardware.ram_total_mb * self.ram_reserve_fraction)

    def _admission_permit(self, waiter: _Waiter) -> ResourcePermit:
        hardware = self.hardware_probe.snapshot()
        reserve = self._effective_ram_reserve(hardware)
        estimate = waiter.job.estimated_ram_mb
        required_headroom = reserve + (estimate or 0.0)
        admission_mode = "ram-aware" if estimate is not None else "serialized-unknown-model-ram"

        if self.ram_admission_enabled and hardware.ram_available_mb < required_headroom:
            estimate_text = f" + {estimate:.0f} MB model estimate" if estimate is not None else ""
            raise ResourceAdmissionError(
                "Local inference admission rejected: "
                f"{hardware.ram_available_mb:.0f} MB RAM available, but "
                f"{reserve:.0f} MB safety reserve{estimate_text} is required."
            )

        return ResourcePermit(
            queue_wait_seconds=round(monotonic() - waiter.enqueued_at, 6),
            queue_depth_at_admission=len(self._waiters),
            admission_mode=admission_mode,
            ram_available_mb=hardware.ram_available_mb,
            ram_reserve_mb=round(reserve, 2),
            estimated_model_ram_mb=estimate,
            cpu_percent=hardware.cpu_percent,
            physical_cores=hardware.physical_cores,
            logical_cores=hardware.logical_cores,
        )

    def _dispatch(self) -> None:
        self._waiters.sort(key=lambda item: (-item.priority, item.sequence))
        while self._active_model < self.max_model_jobs and self._waiters:
            waiter = self._waiters.pop(0)
            if waiter.future.cancelled():
                continue
            try:
                permit = self._admission_permit(waiter)
            except ResourceAdmissionError as exc:
                waiter.future.set_exception(exc)
                continue
            self._active_model += 1
            self._current_model = waiter.job.model
            waiter.future.set_result(permit)

    @asynccontextmanager
    async def acquire_model(self, job: ModelJob) -> AsyncIterator[ResourcePermit]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ResourcePermit] = loop.create_future()
        waiter = _Waiter(job.priority, self._sequence, monotonic(), job, future)
        self._sequence += 1
        self._waiters.append(waiter)
        self._dispatch()
        granted = False
        try:
            permit = await future
            granted = True
            yield permit
        finally:
            if granted:
                self._active_model = max(0, self._active_model - 1)
                self._last_model = job.model
                self._current_model = None
            else:
                self._waiters = [item for item in self._waiters if item is not waiter]
                if not future.done():
                    future.cancel()
            self._dispatch()

    @asynccontextmanager
    async def acquire_cpu(self) -> AsyncIterator[None]:
        await self._cpu_semaphore.acquire()
        self._active_cpu += 1
        try:
            yield
        finally:
            self._active_cpu = max(0, self._active_cpu - 1)
            self._cpu_semaphore.release()

    @contextmanager
    def acquire_cpu_sync(self) -> Iterator[None]:
        """Admission for synchronous CPU pipelines such as local reranking."""
        self._cpu_sync_semaphore.acquire()
        with self._cpu_state_lock:
            self._active_cpu += 1
        try:
            yield
        finally:
            with self._cpu_state_lock:
                self._active_cpu = max(0, self._active_cpu - 1)
            self._cpu_sync_semaphore.release()

    def snapshot(self) -> ResourceSnapshot:
        hardware = self.hardware_probe.snapshot()
        reserve = self._effective_ram_reserve(hardware)
        return ResourceSnapshot(
            max_model_jobs=self.max_model_jobs,
            max_cpu_jobs=self.max_cpu_jobs,
            active_model_jobs=self._active_model,
            active_cpu_jobs=self._active_cpu,
            queue_depth=len(self._waiters),
            current_model=self._current_model,
            last_model=self._last_model,
            cpu_percent=hardware.cpu_percent,
            physical_cores=hardware.physical_cores,
            logical_cores=hardware.logical_cores,
            ram_used_mb=hardware.ram_used_mb,
            ram_available_mb=hardware.ram_available_mb,
            ram_total_mb=hardware.ram_total_mb,
            ram_percent=hardware.ram_percent,
            swap_used_mb=hardware.swap_used_mb,
            swap_total_mb=hardware.swap_total_mb,
            ram_reserve_mb=round(reserve, 2),
            ram_admission_enabled=self.ram_admission_enabled,
            vram_used_mb=hardware.vram_used_mb,
            vram_total_mb=hardware.vram_total_mb,
            vram_source=hardware.vram_source,
            max_gpu_model_jobs=self.max_model_jobs,
            active_gpu_jobs=self._active_model,
        )


@lru_cache(maxsize=1)
def get_resource_scheduler() -> ResourceScheduler:
    settings = get_settings()
    max_model_jobs = settings.model_max_concurrent_jobs or settings.max_gpu_model_jobs
    return ResourceScheduler(
        max_model_jobs=max_model_jobs,
        max_cpu_jobs=settings.max_cpu_jobs,
        ram_admission_enabled=settings.model_ram_admission_enabled,
        ram_reserve_mb=settings.model_ram_reserve_mb,
        ram_reserve_fraction=settings.model_ram_reserve_fraction,
    )
