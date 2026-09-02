from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.workcells.models import WorkcellDefinition, WorkcellStep


@dataclass
class WorkcellHandlerContext:
    task_id: str
    request: str
    definition: WorkcellDefinition
    inputs: dict[str, Any]
    accumulated: dict[str, Any] = field(default_factory=dict)
    services: dict[str, Any] = field(default_factory=dict)


WorkcellHandler = Callable[[WorkcellHandlerContext, WorkcellStep, dict[str, Any]], Awaitable[dict[str, Any]]]


class WorkcellHandlerRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, WorkcellHandler] = {}

    def register(self, name: str, handler: WorkcellHandler) -> None:
        if not name or not name.replace("_", "").isalnum():
            raise ValueError("Handler names must be bounded identifiers")
        if name in self._handlers:
            raise ValueError(f"Duplicate Workcell handler: {name}")
        self._handlers[name] = handler

    def has(self, name: str) -> bool:
        return name in self._handlers

    def get(self, name: str) -> WorkcellHandler:
        if name not in self._handlers:
            raise KeyError(f"Unknown Workcell handler: {name}")
        return self._handlers[name]

    def names(self) -> list[str]:
        return sorted(self._handlers)
