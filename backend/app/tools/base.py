from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolRisk(str, Enum):
    low = "LOW"
    medium = "MEDIUM"
    high = "HIGH"


class ToolResult(BaseModel):
    success: bool
    output: Any = None
    error: str | None = None
    generated_files: list[str] = Field(default_factory=list)


class Tool(ABC):
    name: str
    description: str
    risk: ToolRisk
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    @abstractmethod
    async def execute(self, args: dict[str, Any]) -> ToolResult:
        raise NotImplementedError
