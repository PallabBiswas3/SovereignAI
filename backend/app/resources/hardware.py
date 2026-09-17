from __future__ import annotations

import csv
import shutil
import subprocess

import psutil
from pydantic import BaseModel


class HardwareSnapshot(BaseModel):
    """Point-in-time local hardware state for inference admission and telemetry."""

    cpu_percent: float
    physical_cores: int | None
    logical_cores: int | None
    ram_used_mb: float
    ram_available_mb: float
    ram_total_mb: float
    ram_percent: float
    swap_used_mb: float
    swap_total_mb: float
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    vram_source: str | None = None


class LocalHardwareProbe:
    """Collect lightweight CPU/RAM telemetry without requiring accelerator hardware."""

    def snapshot(self) -> HardwareSnapshot:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        vram_used, vram_total, vram_source = self._vram_snapshot()
        return HardwareSnapshot(
            cpu_percent=round(psutil.cpu_percent(interval=None), 2),
            physical_cores=psutil.cpu_count(logical=False),
            logical_cores=psutil.cpu_count(logical=True),
            ram_used_mb=round(memory.used / 1024 / 1024, 2),
            ram_available_mb=round(memory.available / 1024 / 1024, 2),
            ram_total_mb=round(memory.total / 1024 / 1024, 2),
            ram_percent=round(float(memory.percent), 2),
            swap_used_mb=round(swap.used / 1024 / 1024, 2),
            swap_total_mb=round(swap.total / 1024 / 1024, 2),
            vram_used_mb=vram_used,
            vram_total_mb=vram_total,
            vram_source=vram_source,
        )

    @staticmethod
    def _vram_snapshot() -> tuple[float | None, float | None, str | None]:
        """Optional accelerator telemetry; absence is normal on CPU-only machines."""
        executable = shutil.which("nvidia-smi")
        if not executable:
            return None, None, None
        try:
            result = subprocess.run(
                [
                    executable,
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            rows = list(csv.reader(line for line in result.stdout.splitlines() if line.strip()))
            used = sum(float(row[0].strip()) for row in rows)
            total = sum(float(row[1].strip()) for row in rows)
            return round(used, 2), round(total, 2), "nvidia-smi"
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None, None, None
