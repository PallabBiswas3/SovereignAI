from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from app.router.schemas import TaskProfile


class ExecutionMode(str, Enum):
    automatic = "AUTOMATIC"
    fast = "FAST"
    standard = "STANDARD"
    deep = "DEEP"


class ExecutionModeSelection(BaseModel):
    requested: ExecutionMode
    selected: ExecutionMode
    reason: str
    priority: int


class ExecutionModeSelector:
    """Selects depth without an LLM and keeps the decision auditable."""

    DEEP_PHRASES = {
        "management package",
        "management pack",
        "investigate",
        "root cause",
        "all available evidence",
        "approval note",
        "execute code",
        "run code",
    }
    FAST_PREFIXES = ("what is ", "what are ", "which ", "where is ", "when is ")
    FAST_TERMS = {"limit", "threshold", "value", "definition", "section", "date"}

    def select(
        self,
        requested: ExecutionMode,
        request: str,
        profile: TaskProfile,
        attachment_count: int = 0,
    ) -> ExecutionModeSelection:
        if requested != ExecutionMode.automatic:
            return ExecutionModeSelection(
                requested=requested,
                selected=requested,
                reason=f"User selected {requested.value.lower()} execution.",
                priority=self._priority(requested, profile.task_type),
            )

        lowered = request.lower().strip()
        word_count = len(lowered.split())
        if (
            profile.task_type == "coding"
            or attachment_count > 1
            or any(phrase in lowered for phrase in self.DEEP_PHRASES)
        ):
            selected = ExecutionMode.deep
            reason = "Task requires coding, multiple evidence inputs, or a consequential multi-step deliverable."
        elif (
            attachment_count == 0
            and word_count <= 24
            and lowered.startswith(self.FAST_PREFIXES)
            and any(term in lowered for term in self.FAST_TERMS)
        ):
            selected = ExecutionMode.fast
            reason = "Short factual lookup can use one bounded generation or retrieval path."
        else:
            selected = ExecutionMode.standard
            reason = "Normal analysis benefits from transparent preparation and verification."
        return ExecutionModeSelection(
            requested=requested,
            selected=selected,
            reason=reason,
            priority=self._priority(selected, profile.task_type),
        )

    @staticmethod
    def _priority(mode: ExecutionMode, task_type: str) -> int:
        base = {
            ExecutionMode.fast: 30,
            ExecutionMode.standard: 50,
            ExecutionMode.deep: 70,
            ExecutionMode.automatic: 50,
        }[mode]
        return min(100, base + (10 if task_type == "coding" else 0))

