from __future__ import annotations

import asyncio
import csv
import shutil
import subprocess
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from time import monotonic
from typing import AsyncIterator
from typing import Iterator
from contextlib import contextmanager
from threading import BoundedSemaphore, Lock

import psutil
from pydantic import BaseModel, Field

from app.core.config import get_settings


class ModelJob(BaseModel):
    model: str
    role: str = "GENERAL"
    memory_requirement: str = "medium"
    execution_mode: str = "STANDARD"
    priority: int = Field(default=50, ge=0, le=100)


class ResourcePermit(BaseModel):
    queue_wait_seconds: float
    queue_depth_at_admission: int


class ResourceSnapshot(BaseModel):
    max_gpu_model_jobs: int
    max_cpu_jobs: int
    active_gpu_jobs: int
    active_cpu_jobs: int
    queue_depth: int
    current_model: str | None
    last_model: str | None
    cpu_percent: float
    ram_used_mb: float
    ram_total_mb: float
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    vram_source: str | None = None


@dataclass(slots=True)
class _Waiter:
    priority: int
    sequence: int
    enqueued_at: float
    job: ModelJob
    future: asyncio.Future[ResourcePermit]


class ResourceScheduler:
    """Small, process-local admission controller for memory-constrained inference.

    It serializes local generative-model work by default. It does not claim to
    unload models or control GPU memory; Ollama remains responsible for that.
    """

    def __init__(self, max_gpu_model_jobs: int = 1, max_cpu_jobs: int = 2) -> None:
        self.max_gpu_model_jobs = max(1, max_gpu_model_jobs)
        self.max_cpu_jobs = max(1, max_cpu_jobs)
        self._active_gpu = 0
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

    def _dispatch(self) -> None:
        self._waiters.sort(key=lambda item: (-item.priority, item.sequence))
        while self._active_gpu < self.max_gpu_model_jobs and self._waiters:
            waiter = self._waiters.pop(0)
            if waiter.future.cancelled():
                continue
            self._active_gpu += 1
            self._current_model = waiter.job.model
            waiter.future.set_result(ResourcePermit(
                queue_wait_seconds=round(monotonic() - waiter.enqueued_at, 6),
                queue_depth_at_admission=len(self._waiters),
            ))

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
                self._active_gpu = max(0, self._active_gpu - 1)
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
        memory = psutil.virtual_memory()
        vram_used, vram_total, source = self._vram_snapshot()
        return ResourceSnapshot(
            max_gpu_model_jobs=self.max_gpu_model_jobs,
            max_cpu_jobs=self.max_cpu_jobs,
            active_gpu_jobs=self._active_gpu,
            active_cpu_jobs=self._active_cpu,
            queue_depth=len(self._waiters),
            current_model=self._current_model,
            last_model=self._last_model,
            cpu_percent=round(psutil.cpu_percent(interval=None), 2),
            ram_used_mb=round(memory.used / 1024 / 1024, 2),
            ram_total_mb=round(memory.total / 1024 / 1024, 2),
            vram_used_mb=vram_used,
            vram_total_mb=vram_total,
            vram_source=source,
        )

    @staticmethod
    def _vram_snapshot() -> tuple[float | None, float | None, str | None]:
        executable = shutil.which("nvidia-smi")
        if not executable:
            return None, None, None
        try:
            result = subprocess.run(
                [
                    executable,
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            rows = list(csv.reader(line for line in result.stdout.splitlines() if line.strip()))
            used = sum(float(row[0].strip()) for row in rows)
            total = sum(float(row[1].strip()) for row in rows)
            return round(used, 2), round(total, 2), "nvidia-smi"
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None, None, None


@lru_cache(maxsize=1)
def get_resource_scheduler() -> ResourceScheduler:
    settings = get_settings()
    return ResourceScheduler(settings.max_gpu_model_jobs, settings.max_cpu_jobs)
