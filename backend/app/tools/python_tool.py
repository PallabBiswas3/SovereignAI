from __future__ import annotations

from typing import Any

from app.sandbox.executor import DockerSandboxExecutor
from app.tools.base import Tool, ToolResult, ToolRisk
from app.tools.file_tools import SafeWorkspace


class PythonSandboxTool(Tool):
    name = "run_python"
    description = "Execute Python inside a resource-limited, networkless Docker container"
    risk = ToolRisk.medium
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "input_files": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["code"],
    }

    def __init__(self, executor: DockerSandboxExecutor, workspace: SafeWorkspace | None = None) -> None:
        self.executor = executor
        self.workspace = workspace

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        code = str(args.get("code", ""))
        if not code.strip():
            return ToolResult(success=False, error="Python code is required")
        inputs = []
        for value in args.get("input_files", []):
            if not self.workspace:
                return ToolResult(success=False, error="Input files are unavailable for this tool instance")
            try:
                inputs.append(str(self.workspace.resolve(str(value), must_exist=True)))
            except (ValueError, FileNotFoundError) as exc:
                return ToolResult(success=False, error=str(exc))
        result = await self.executor.execute(code, inputs)
        return ToolResult(success=result.executed and result.exit_code == 0, output=result.model_dump(), error=None if result.executed else result.stderr, generated_files=result.generated_files)
