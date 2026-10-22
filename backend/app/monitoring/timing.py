"""Request-local latency measurements. Never retain prompts or model output."""
from __future__ import annotations

import asyncio
from contextlib import aclosing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import math
from time import perf_counter
from typing import Any

from app.llm.base import ModelGenerationCancelled


_current: ContextVar[TaskTiming | None] = ContextVar("task_timing", default=None)
_purpose: ContextVar[str | None] = ContextVar("inference_purpose", default=None)
_STAT_KEYS = (
    "prompt_token_count", "token_count", "prompt_processing_seconds",
    "generation_seconds", "load_duration_seconds", "queue_wait_seconds",
    "time_to_first_token_seconds", "tokens_per_second", "total_duration_seconds",
)


def _number(value: Any) -> float | int | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and value >= 0:
            return value
    return None


@dataclass
class TaskTiming:
    started: float = field(default_factory=perf_counter)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    stages: dict[str, float] = field(default_factory=dict)

    def record_call(self, model: str, provider: str, started: float,
                    outcome: str, stats: dict[str, Any], kind: str) -> None:
        self.llm_calls.append({
            "call": len(self.llm_calls) + 1, "model": model, "provider": provider,
            "purpose": _purpose.get() or kind, "outcome": outcome,
            "wall_seconds": round(perf_counter() - started, 6),
            "warm_status": stats.get("warm_status") if stats.get("warm_status") in {"warm", "cold"} else "unknown",
            **{key: _number(stats.get(key)) for key in _STAT_KEYS},
        })

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task_handler_seconds": round(perf_counter() - self.started, 6),
            "llm_call_count": len(self.llm_calls),
            "llm_wall_seconds": round(sum(call["wall_seconds"] for call in self.llm_calls), 6),
            "tool_seconds": round(sum(tool["wall_seconds"] for tool in self.tools), 6),
            "stages_seconds": dict(self.stages),
            "llm_calls": list(self.llm_calls), "tools": list(self.tools),
            "scope": "Task handler through audit persistence. Final timing snapshot write, SSE delivery and browser rendering excluded.",
            "coverage": "Local Ollama/vLLM text-provider calls and bounded-agent tools. Vision, external services and other workflow steps are included in execution time but not individually instrumented.",
        }


@contextmanager
def capture_task_timing():
    timing = TaskTiming()
    token = _current.set(timing)
    try:
        yield timing
    finally:
        _current.reset(token)


@contextmanager
def inference_purpose(purpose: str):
    token = _purpose.set(purpose)
    try:
        yield
    finally:
        _purpose.reset(token)


def record_stage(name: str, started: float) -> None:
    timing = _current.get()
    if timing is not None:
        timing.stages[name] = round(perf_counter() - started, 6)


def record_tool(tool: str, success: bool, seconds: float) -> None:
    timing = _current.get()
    if timing is not None:
        timing.tools.append({"tool": tool, "success": success, "wall_seconds": round(seconds, 6)})


def timing_snapshot() -> dict[str, Any] | None:
    timing = _current.get()
    return timing.snapshot() if timing is not None else None


def measure_generation(method):
    """Measure a structured call once, including queueing and failures."""
    @wraps(method)
    async def measured(self, prompt, model, *args, **kwargs):
        timing = _current.get()
        if timing is None:
            return await method(self, prompt, model, *args, **kwargs)
        started, stats, outcome = perf_counter(), {}, "failed"
        provider = type(self).__name__
        try:
            result = await method(self, prompt, model, *args, **kwargs)
            stats = result.runtime_stats
            outcome = "fallback" if result.fallback else "completed"
            return result
        except (asyncio.CancelledError, ModelGenerationCancelled):
            outcome = "cancelled"
            raise
        finally:
            timing.record_call(model, provider, started, outcome, stats, "structured_generation")
    return measured


def measure_stream(method):
    """Measure the stream, not the generate() method that consumes it."""
    @wraps(method)
    async def measured(self, prompt, model, *args, **kwargs):
        timing = _current.get()
        started, stats, outcome = perf_counter(), {}, "failed"
        try:
            async with aclosing(method(self, prompt, model, *args, **kwargs)) as stream:
                async for chunk in stream:
                    if chunk.done:
                        stats = chunk.runtime_stats
                        outcome = "fallback" if chunk.fallback else "completed"
                    yield chunk
        except (asyncio.CancelledError, ModelGenerationCancelled, GeneratorExit):
            outcome = "cancelled"
            raise
        finally:
            if timing is not None:
                timing.record_call(model, type(self).__name__, started, outcome, stats, "streamed_generation")
    return measured
