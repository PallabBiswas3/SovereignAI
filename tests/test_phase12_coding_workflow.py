import asyncio
from pathlib import Path

from app.sandbox.executor import SandboxResult
from app.llm.base import StructuredGenerationResult
from app.workflows.coding import CodingWorkflow


class FakeSandbox:
    async def execute(self, code: str, input_files: list[str] | None = None) -> SandboxResult:
        assert "--z-threshold" in code
        assert input_files
        return SandboxResult(exit_code=0, stdout='{"rows": 8, "anomalies": 2}', executed=True, isolation="docker")


def test_coding_workflow_produces_source_and_report(tmp_path: Path) -> None:
    csv_path = tmp_path / "pump_sensor_readings.csv"
    csv_path.write_text("timestamp,temperature_c,vibration_mm_s\n1,72,3.2\n2,91,9.6\n", encoding="utf-8")
    result = asyncio.run(CodingWorkflow(FakeSandbox()).run(csv_path, tmp_path / "artifacts"))
    assert Path(result.source_path).is_file()
    assert Path(result.report_path).is_file()
    assert result.execution["isolation"] == "docker"


class RepairingProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_json(self, prompt, model, schema, system=None):
        self.calls += 1
        code = "raise RuntimeError('first attempt')" if self.calls == 1 else "print('verified repair')"
        return StructuredGenerationResult(
            text="", data={"code": code, "summary": "bounded test", "expected_outputs": []},
            model=model, provider="test",
        )


class RepairingSandbox:
    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, code: str, input_files=None) -> SandboxResult:
        self.calls += 1
        if self.calls == 1:
            return SandboxResult(exit_code=1, stderr="RuntimeError: first attempt", executed=True)
        return SandboxResult(exit_code=0, stdout="verified repair", executed=True)


def test_coder_model_repairs_failed_code_with_bounded_attempts(tmp_path: Path) -> None:
    csv_path = tmp_path / "readings.csv"
    csv_path.write_text("time,value\n1,4\n2,9\n", encoding="utf-8")
    provider = RepairingProvider()
    sandbox = RepairingSandbox()
    result = asyncio.run(CodingWorkflow(sandbox, provider, "qwen2.5-coder:7b").run(
        csv_path, tmp_path / "artifacts", "Find unusual values", "run-1"
    ))
    assert len(result.attempts) == 2
    assert result.attempts[0].verified is False
    assert result.attempts[1].verified is True
    assert result.model_used == "qwen2.5-coder:7b"
    assert result.used_deterministic_fallback is False
    assert provider.calls == 2
