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
        # Docker's volume argument uses host syntax, not Win32 extended I/O syntax.
        mount_path = str(run_dir)
        if mount_path.startswith("\\\\?\\UNC\\"):
            mount_path = "\\\\" + mount_path[8:]
        elif mount_path.startswith("\\\\?\\"):
            mount_path = mount_path[4:]
        container_name = f"sovereign-sandbox-{run_id}"
        command = [
            "docker", "run", "--rm", "--name", container_name, "--network", "none", "--memory", "256m",
            "--cpus", "0.5", "--pids-limit", "64", "--read-only", "--security-opt",
            "no-new-privileges", "--cap-drop", "ALL", "--user", "65534:65534",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=32m", "-v", f"{mount_path}:/work:rw",
            "-w", "/work", self.image, "python", "-I", "main.py",
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
            input_names = {Path(item).name for item in input_files or []}
            # Only regular files are outputs. A symlink written by sandboxed code
            # into the bind mount would resolve against the host filesystem.
            files = [path.name for path in run_dir.iterdir()
                     if path.name != "main.py" and path.name not in input_names
                     and not path.is_symlink() and path.is_file()]
            return SandboxResult(exit_code=process.returncode, stdout=stdout.decode(errors="replace")[:100_000], stderr=stderr.decode(errors="replace")[:100_000], generated_files=files, executed=True, output_directory=str(run_dir))
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            # Killing the docker CLI does not stop the container (SIGKILL is not
            # proxied), so remove the named container explicitly.
            cleanup = await self._remove_container(container_name)
            return SandboxResult(exit_code=None, stderr=f"Execution exceeded {self.timeout}s timeout{cleanup}", executed=True, output_directory=str(run_dir))
        except OSError as exc:
            return SandboxResult(exit_code=None, stderr=f"Docker execution unavailable: {exc}")

    @staticmethod
    async def _remove_container(name: str) -> str:
        try:
            remover = await asyncio.create_subprocess_exec(
                "docker", "rm", "--force", name,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(remover.wait(), timeout=30)
        except (OSError, asyncio.TimeoutError):
            return "; container cleanup could not be confirmed"
        return "" if remover.returncode == 0 else "; container cleanup could not be confirmed"
