from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.sandbox.executor import DockerSandboxExecutor
from app.tools.base import Tool
from app.tools.file_tools import FileMetadataTool, ListFilesTool, ReadFileTool, SafeWorkspace, SearchFilesTool, WriteFileTool
from app.tools.python_tool import PythonSandboxTool

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from app.core.config import Settings
    from app.identity.models import Principal, ResourceScope


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def validate_arguments(self, name: str, arguments: dict[str, object]) -> list[str]:
        """Apply the small JSON-schema subset used by registered tools."""
        schema = self.get(name).input_schema
        errors: list[str] = []
        for required in schema.get("required", []):
            if required not in arguments:
                errors.append(f"Missing required argument: {required}")
        properties = schema.get("properties", {})
        for key, value in arguments.items():
            expected = properties.get(key, {}).get("type") if isinstance(properties, dict) else None
            valid = {
                "string": isinstance(value, str),
                "integer": isinstance(value, int) and not isinstance(value, bool),
                "array": isinstance(value, list),
                "object": isinstance(value, dict),
                "boolean": isinstance(value, bool),
            }.get(expected, True)
            if not valid:
                errors.append(f"Argument {key} must be {expected}")
        return errors

    def discover(self) -> list[dict[str, object]]:
        return [{"name": tool.name, "description": tool.description, "risk": tool.risk.value,
                 "input_schema": tool.input_schema} for tool in self._tools.values()]


def create_default_registry(workspace_root: Path) -> ToolRegistry:
    workspace = SafeWorkspace(workspace_root)
    registry = ToolRegistry()
    for tool in (
        ListFilesTool(workspace), ReadFileTool(workspace), WriteFileTool(workspace),
        SearchFilesTool(workspace), FileMetadataTool(workspace),
        PythonSandboxTool(DockerSandboxExecutor(workspace_root / "sandbox"), workspace),
    ):
        registry.register(tool)
    return registry


def create_agent_registry(
    settings: "Settings", session: "Session", principal: "Principal | None" = None,
    scope: "ResourceScope | None" = None,
) -> ToolRegistry:
    from app.artifacts.service import ArtifactService
    from app.router.model_registry import ModelRegistry
    from app.tools.application_tools import (
        AnalyzeImageTool,
        GenerateDocxTool,
        GeneratePptxTool,
        GenerateXlsxTool,
        KnowledgeSearchTool,
        OCRDocumentTool,
    )

    registry = create_default_registry(settings.workspace_root)
    workspace = SafeWorkspace(settings.workspace_root)
    artifacts = ArtifactService(session, settings.workspace_root / "artifacts")
    vision = ModelRegistry(settings.models_config).get("vision")
    for tool in (
        KnowledgeSearchTool(session, principal), OCRDocumentTool(workspace),
        AnalyzeImageTool(workspace, vision.endpoint, vision.model_tag),
        GenerateDocxTool(artifacts, scope), GenerateXlsxTool(artifacts, scope),
        GeneratePptxTool(artifacts, scope),
    ):
        registry.register(tool)
    return registry
