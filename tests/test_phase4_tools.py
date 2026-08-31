import asyncio
from pathlib import Path

from app.sandbox.executor import DockerSandboxExecutor
from app.tools.file_tools import ReadFileTool, SafeWorkspace, SearchFilesTool, WriteFileTool


def test_file_tools_are_workspace_confined(tmp_path: Path) -> None:
    workspace = SafeWorkspace(tmp_path / "workspace")
    writer = WriteFileTool(workspace)
    reader = ReadFileTool(workspace)
    search = SearchFilesTool(workspace)

    written = asyncio.run(writer.execute({"path": "reports/note.md", "content": "Pump-102 vibration 8.2 mm/s"}))
    read = asyncio.run(reader.execute({"path": "reports/note.md"}))
    found = asyncio.run(search.execute({"query": "vibration"}))
    escaped = asyncio.run(reader.execute({"path": "../../outside.txt"}))

    assert written.success
    assert "8.2 mm/s" in read.output["text"]
    assert found.output[0]["file"] == "reports/note.md"
    assert not escaped.success
    assert "traversal" in escaped.error.lower()


def test_sandbox_never_falls_back_to_host_execution(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("app.sandbox.executor.shutil.which", lambda _: None)
    result = asyncio.run(DockerSandboxExecutor(tmp_path).execute("print('must not run')"))
    assert result.executed is False
    assert result.exit_code is None
    assert "not executed on the host" in result.stderr

