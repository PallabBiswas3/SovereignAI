from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field


class SandboxResult(BaseModel):
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    generated_files: list[str] = Field(default_factory=list)
    executed: bool = False
    isolation: str = "docker"
    output_directory: str | None = None


class DockerSandboxExecutor:
    """Runs generated Python only in an unprivileged, networkless Docker container."""

    def __init__(self, root: Path, image: str = "sovereign-sandbox:py311", timeout: int = 20) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.image = image
        self.timeout = timeout

    async def execute(self, code: str, input_files: list[str] | None = None) -> SandboxResult:
        if not shutil.which("docker"):
            return SandboxResult(exit_code=None, stderr="Docker CLI unavailable; code was not executed on the host.")
        run_id = str(uuid4())
        run_dir = (self.root / run_id).resolve()
        if self.root not in run_dir.parents:
            return SandboxResult(exit_code=None, stderr="Invalid sandbox path")
        run_dir.mkdir(parents=True)
        (run_dir / "main.py").write_text(code, encoding="utf-8")
        for item in input_files or []:
            source = Path(item).resolve()
            if not source.is_file():
                return SandboxResult(exit_code=None, stderr=f"Sandbox input file is missing: {source.name}")
            shutil.copy2(source, run_dir / source.name)
        command = [
            "docker", "run", "--rm", "--network", "none", "--memory", "256m",
            "--cpus", "0.5", "--pids-limit", "64", "--read-only", "--security-opt",
            "no-new-privileges", "--cap-drop", "ALL", "--user", "65534:65534",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=32m", "-v", f"{run_dir}:/work:rw",
            "-w", "/work", self.image, "python", "-I", "main.py",
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
            input_names = {Path(item).name for item in input_files or []}
            files = [path.name for path in run_dir.iterdir() if path.name != "main.py" and path.name not in input_names]
            return SandboxResult(exit_code=process.returncode, stdout=stdout.decode(errors="replace")[:100_000], stderr=stderr.decode(errors="replace")[:100_000], generated_files=files, executed=True, output_directory=str(run_dir))
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return SandboxResult(exit_code=None, stderr=f"Execution exceeded {self.timeout}s timeout", executed=True, output_directory=str(run_dir))
        except OSError as exc:
            return SandboxResult(exit_code=None, stderr=f"Docker execution unavailable: {exc}")
