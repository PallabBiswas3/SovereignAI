from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from threading import RLock

from pydantic import BaseModel


class ModelLifecycleState(str, Enum):
    not_installed = "NOT_INSTALLED"
    cold = "COLD"
    loading = "LOADING"
    ready = "READY"
    busy = "BUSY"
    idle = "IDLE"
    error = "ERROR"


class ModelLifecycleSnapshot(BaseModel):
    model: str
    state: ModelLifecycleState
    last_used_at: datetime | None = None
    load_duration_seconds: float | None = None
    time_to_first_token_seconds: float | None = None
    tokens_per_second: float | None = None
    generation_duration_seconds: float | None = None
    token_count: int | None = None
    failure_count: int = 0
    attempt_count: int = 0
    failure_rate: float = 0.0
    memory_usage_mb: float | None = None
    queue_depth: int = 0
    warm_status: str = "unknown"
    last_error: str | None = None


class ModelLifecycleManager:
    """Tracks observed process-local model state without claiming GPU control."""

    def __init__(self) -> None:
        self._records: dict[str, ModelLifecycleSnapshot] = {}
        self._lock = RLock()

    def _record(self, model: str) -> ModelLifecycleSnapshot:
        return self._records.setdefault(
            model,
            ModelLifecycleSnapshot(model=model, state=ModelLifecycleState.cold),
        )

    def observe(
        self,
        model: str,
        *,
        installed: bool,
        loaded: bool,
        memory_usage_mb: float | None = None,
    ) -> ModelLifecycleSnapshot:
        with self._lock:
            record = self._record(model)
            if record.state not in {ModelLifecycleState.busy, ModelLifecycleState.loading}:
                record.state = (
                    ModelLifecycleState.not_installed
                    if not installed
                    else ModelLifecycleState.ready
                    if loaded and record.last_used_at is None
                    else ModelLifecycleState.idle
                    if loaded
                    else ModelLifecycleState.cold
                )
            record.memory_usage_mb = memory_usage_mb
            record.warm_status = "warm" if loaded else "cold" if installed else "unavailable"
            return record.model_copy(deep=True)

    def begin(self, model: str, *, was_loaded: bool | None, queue_depth: int) -> None:
        with self._lock:
            record = self._record(model)
            record.attempt_count += 1
            record.queue_depth = queue_depth
            record.warm_status = "warm" if was_loaded is True else "cold" if was_loaded is False else "unknown"
            record.state = ModelLifecycleState.busy if was_loaded else ModelLifecycleState.loading
            record.last_error = None

    def mark_busy(self, model: str) -> None:
        with self._lock:
            self._record(model).state = ModelLifecycleState.busy

    def complete(self, model: str, stats: dict[str, object]) -> None:
        with self._lock:
            record = self._record(model)
            record.state = ModelLifecycleState.idle
            record.last_used_at = datetime.now(timezone.utc)
            record.load_duration_seconds = _optional_float(stats.get("load_duration_seconds"))
            record.time_to_first_token_seconds = _optional_float(stats.get("time_to_first_token_seconds"))
            record.tokens_per_second = _optional_float(stats.get("tokens_per_second"))
            record.generation_duration_seconds = _optional_float(stats.get("total_duration_seconds"))
            record.token_count = _optional_int(stats.get("token_count"))
            record.queue_depth = 0
            record.failure_rate = round(record.failure_count / max(1, record.attempt_count), 4)

    def fail(self, model: str, error: str) -> None:
        with self._lock:
            record = self._record(model)
            record.state = ModelLifecycleState.error
            record.last_used_at = datetime.now(timezone.utc)
            record.failure_count += 1
            record.failure_rate = round(record.failure_count / max(1, record.attempt_count), 4)
            record.last_error = error
            record.queue_depth = 0

    def get(self, model: str) -> ModelLifecycleSnapshot:
        with self._lock:
            return self._record(model).model_copy(deep=True)

    def all(self) -> list[ModelLifecycleSnapshot]:
        with self._lock:
            return [record.model_copy(deep=True) for record in self._records.values()]


def _optional_float(value: object) -> float | None:
    return round(float(value), 6) if isinstance(value, (int, float)) else None


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


@lru_cache(maxsize=1)
def get_model_lifecycle_manager() -> ModelLifecycleManager:
    return ModelLifecycleManager()

