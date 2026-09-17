from __future__ import annotations

from abc import ABC, abstractmethod
from ..models import AgentResult, SignalRecord


class DiagnosticAgent(ABC):
    name: str = "diagnostic-agent"

    @abstractmethod
    def run(self, record: SignalRecord, context: dict | None = None) -> AgentResult:
        raise NotImplementedError
