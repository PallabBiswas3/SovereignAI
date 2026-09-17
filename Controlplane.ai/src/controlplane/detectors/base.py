from __future__ import annotations

import abc
from time import perf_counter
from typing import Any

from controlplane.schema import DetectorResult, Interaction


class Detector(abc.ABC):
    name: str
    version: str = "1.0"

    def run(self, interaction: Interaction, settings: dict[str, Any] | None = None) -> DetectorResult:
        started = perf_counter()
        try:
            result = self.detect(interaction, settings or {})
            result.latency_ms = (perf_counter() - started) * 1000.0
            return result
        except Exception as exc:  # isolate detector failure from the safety gateway
            return DetectorResult(
                detector=self.name,
                latency_ms=(perf_counter() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )

    @abc.abstractmethod
    def detect(self, interaction: Interaction, settings: dict[str, Any]) -> DetectorResult:
        raise NotImplementedError
