"""
Wall-clock timing utilities.

Per the correction in chat: latency must be measured as real wall-clock
time around COMPLETE operations (including network, serialization,
reranking, tool calls -- everything), not framework-reported inference
time. Otherwise it's easy to accidentally report "verification added
only 20ms" when a job was actually dispatched asynchronously and took
2 seconds to complete, which would invalidate the whole latency
analysis this project is built around.

Usage:
    with Timer() as t:
        do_something()
    print(t.elapsed_ms)

    @timed
    def do_something(...): ...
    result, elapsed_ms = do_something(...)
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Timer:
    """Context manager that records wall-clock elapsed time in milliseconds
    using time.perf_counter (monotonic, unaffected by system clock changes —
    appropriate for latency measurement, unlike time.time())."""

    def __init__(self) -> None:
        self.start_ns: int | None = None
        self.end_ns: int | None = None

    def __enter__(self) -> "Timer":
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.end_ns = time.perf_counter_ns()

    @property
    def elapsed_ms(self) -> float:
        if self.start_ns is None:
            raise RuntimeError("Timer never started")
        end = self.end_ns if self.end_ns is not None else time.perf_counter_ns()
        return (end - self.start_ns) / 1_000_000


def timed(func: Callable[..., T]) -> Callable[..., tuple[T, float]]:
    """Decorator: wraps `func` so calling it returns (result, elapsed_ms)
    instead of just the result. Useful for quickly instrumenting a
    pipeline stage without rewriting it as a class."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[T, float]:
        with Timer() as t:
            result = func(*args, **kwargs)
        return result, t.elapsed_ms

    return wrapper
