from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.llm.base import LocalModelProvider
from app.sandbox.executor import DockerSandboxExecutor, SandboxResult


MAX_CODE_REPAIR_ATTEMPTS = 3

ANOMALY_SCRIPT = '''#!/usr/bin/env python3
"""Emergency deterministic pump sensor anomaly detector."""
from __future__ import annotations
import argparse, csv, json, statistics
from pathlib import Path

def analyze(source: Path, output: Path, z_threshold: float = 2.0) -> dict[str, object]:
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows: raise ValueError("CSV contains no data rows")
    numeric = [name for name in ("temperature_c", "vibration_mm_s") if name in rows[0]]
    if not numeric: raise ValueError("Expected temperature_c and/or vibration_mm_s column")
    stats = {name: (statistics.mean(float(row[name]) for row in rows), statistics.pstdev(float(row[name]) for row in rows)) for name in numeric}
    anomalies = []
    for row in rows:
        reasons = []
        for name in numeric:
            value = float(row[name]); mean, deviation = stats[name]
            z_score = abs(value - mean) / deviation if deviation else 0.0
            limit = 80.0 if name == "temperature_c" else 6.0
            if z_score >= z_threshold or value > limit: reasons.append(f"{name}: value={value:g}, z={z_score:.2f}, limit={limit:g}")
        if reasons: anomalies.append({**row, "anomaly_reasons": "; ".join(reasons)})
    fields = list(rows[0]) + ["anomaly_reasons"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(anomalies)
    return {"rows": len(rows), "anomalies": len(anomalies), "output": str(output)}

def main() -> None:
    default_input = next(Path(".").glob("*.csv"), None)
    parser = argparse.ArgumentParser(); parser.add_argument("input", nargs="?", default=str(default_input) if default_input else "input.csv")
    parser.add_argument("--output", default="anomalies.csv"); parser.add_argument("--z-threshold", type=float, default=2.0)
    args = parser.parse_args(); summary = analyze(Path(args.input), Path(args.output), args.z_threshold)
    Path("analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps(summary))
if __name__ == "__main__": main()
'''


class DatasetProfile(BaseModel):
    filename: str
    row_count: int
    columns: list[str]
    inferred_types: dict[str, str]
    missing_values: dict[str, int]
    sample_rows: list[dict[str, str]]


class CodeAttempt(BaseModel):
    attempt: int
    version: str
    model_used: str
    exit_code: int | None
    executed: bool
    stdout: str
    stderr_summary: str
    verified: bool


class CodingWorkflowResult(BaseModel):
    source_path: str
    report_path: str
    result_path: str | None = None
    result_paths: list[str] = Field(default_factory=list)
    execution: dict[str, object]
    attempts: list[CodeAttempt] = Field(default_factory=list)
    dataset_profile: DatasetProfile
    model_used: str
    used_deterministic_fallback: bool = False
    warnings: list[str] = Field(default_factory=list)


class CodingWorkflow:
    def __init__(
        self,
        executor: DockerSandboxExecutor,
        provider: LocalModelProvider | None = None,
        model_tag: str = "deterministic-fallback",
        max_repair_attempts: int = MAX_CODE_REPAIR_ATTEMPTS,
    ) -> None:
        self.executor = executor
        self.provider = provider
        self.model_tag = model_tag
        self.max_repair_attempts = max(0, min(max_repair_attempts, MAX_CODE_REPAIR_ATTEMPTS))

    async def run(
        self, csv_path: Path, artifact_root: Path, request: str = "Analyze anomalies", run_id: str | None = None
    ) -> CodingWorkflowResult:
        profile = self.profile_csv(csv_path)
        run_root = artifact_root / (run_id or "coding-run")
        run_root.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        used_fallback = self.provider is None
        code = ANOMALY_SCRIPT
        summary = "Emergency deterministic anomaly analysis."
        if self.provider:
            generated = await self.provider.generate_json(
                self._generation_prompt(request, profile), self.model_tag, self._response_schema(),
                "You are a local coding agent. Return concise structured output, never shell commands or markdown fences.",
            )
            if generated.fallback or not generated.data or not str(generated.data.get("code", "")).strip():
                used_fallback = True
                warnings.append("Coder model was unavailable or returned invalid code; deterministic emergency template used.")
            else:
                code = self._clean_code(str(generated.data["code"]))
                summary = str(generated.data.get("summary", "Coder model generated the analysis program."))

        attempts: list[CodeAttempt] = []
        execution = SandboxResult(exit_code=None, stderr="Code was not executed.")
        total_attempts = 1 + (0 if used_fallback else self.max_repair_attempts)
        for attempt_number in range(1, total_attempts + 1):
            version_name = f"pump_anomaly_detector_v{attempt_number}.py"
            (run_root / version_name).write_text(code, encoding="utf-8")
            execution = await self.executor.execute(code, [str(csv_path)])
            verified = bool(execution.executed and execution.exit_code == 0)
            attempts.append(CodeAttempt(
                attempt=attempt_number, version=version_name, model_used=self.model_tag if not used_fallback else "deterministic-fallback",
                exit_code=execution.exit_code, executed=execution.executed,
                stdout=execution.stdout[:4_000], stderr_summary=execution.stderr[:4_000], verified=verified,
            ))
            if verified or not execution.executed or used_fallback or attempt_number >= total_attempts:
                break
            repaired = await self.provider.generate_json(
                self._repair_prompt(request, profile, code, execution, attempt_number),
                self.model_tag, self._response_schema(),
                "Repair only the Python program using the observed error. Return structured JSON without markdown fences.",
            ) if self.provider else None
            if not repaired or repaired.fallback or not repaired.data or not str(repaired.data.get("code", "")).strip():
                warnings.append("Coder model could not produce a valid repair; retry loop stopped safely.")
                break
            code = self._clean_code(str(repaired.data["code"]))
            summary = str(repaired.data.get("summary", summary))

        source_path = run_root / "pump_anomaly_detector.py"
        source_path.write_text(code, encoding="utf-8")
        result_paths = self._collect_outputs(execution, csv_path, run_root)
        verified = bool(execution.executed and execution.exit_code == 0)
        if not verified:
            warnings.append(execution.stderr or "Sandbox execution did not complete successfully.")
        report_path = run_root / "coding_analysis_report.md"
        report_path.write_text(self._report(request, profile, summary, attempts, verified), encoding="utf-8")
        csv_results = [path for path in result_paths if Path(path).suffix.lower() == ".csv"]
        return CodingWorkflowResult(
            source_path=str(source_path), report_path=str(report_path),
            result_path=csv_results[0] if csv_results else None, result_paths=result_paths,
            execution=execution.model_dump(), attempts=attempts, dataset_profile=profile,
            model_used=self.model_tag if not used_fallback else "deterministic-fallback",
            used_deterministic_fallback=used_fallback, warnings=list(dict.fromkeys(warnings)),
        )

    @staticmethod
    def profile_csv(path: Path, sample_limit: int = 5) -> DatasetProfile:
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            rows = list(reader)
        if not columns:
            raise ValueError("CSV has no header row")
        missing = {column: sum(not str(row.get(column, "")).strip() for row in rows) for column in columns}
        inferred = {column: CodingWorkflow._infer_type([str(row.get(column, "")) for row in rows]) for column in columns}
        samples = [{column: str(row.get(column, ""))[:200] for column in columns} for row in rows[:sample_limit]]
        return DatasetProfile(
            filename=path.name, row_count=len(rows), columns=columns,
            inferred_types=inferred, missing_values=missing, sample_rows=samples,
        )

    @staticmethod
    def _infer_type(values: list[str]) -> str:
        present = [value.strip() for value in values if value.strip()]
        if not present:
            return "empty"
        try:
            for value in present: int(value)
            return "integer"
        except ValueError:
            pass
        try:
            for value in present: float(value)
            return "float"
        except ValueError:
            return "string"

    @staticmethod
    def _response_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "summary": {"type": "string"},
                "expected_outputs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["code", "summary", "expected_outputs"],
        }

    @staticmethod
    def _generation_prompt(request: str, profile: DatasetProfile) -> str:
        return (
            "Create a complete Python 3.11 program for this requirement:\n" + request +
            "\nThe program runs in a networkless container with only the Python standard library. "
            "Read the attached CSV by its exact filename, do not invent columns, print a concise JSON summary, "
            "and write useful analysis output such as analysis_summary.json and a result CSV. "
            "Never access the network, environment secrets, parent paths, or invoke subprocesses.\n"
            "Dataset profile:\n" + profile.model_dump_json(indent=2)
        )

    @staticmethod
    def _repair_prompt(
        request: str, profile: DatasetProfile, code: str, execution: SandboxResult, attempt: int
    ) -> str:
        return (
            f"Repair attempt {attempt}. Original requirement:\n{request}\nDataset profile:\n"
            f"{profile.model_dump_json(indent=2)}\nCurrent code:\n{code[:30_000]}\n"
            f"Observed exit code: {execution.exit_code}\nstdout:\n{execution.stdout[:4_000]}\n"
            f"stderr:\n{execution.stderr[:8_000]}\nReturn corrected complete code."
        )

    @staticmethod
    def _clean_code(code: str) -> str:
        stripped = code.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            lines = lines[1:] if lines else lines
            if lines and lines[-1].strip() == "```": lines = lines[:-1]
            stripped = "\n".join(lines)
        return stripped

    @staticmethod
    def _collect_outputs(execution: SandboxResult, input_path: Path, destination: Path) -> list[str]:
        if not execution.output_directory:
            return []
        source_root = Path(execution.output_directory)
        allowed = {".csv", ".json", ".png", ".jpg", ".jpeg", ".svg", ".txt", ".md"}
        collected = []
        for name in execution.generated_files:
            source = source_root / Path(name).name
            if source.name == input_path.name or source.suffix.lower() not in allowed or not source.is_file():
                continue
            target = destination / source.name
            shutil.copy2(source, target)
            collected.append(str(target))
        return collected

    @staticmethod
    def _report(
        request: str, profile: DatasetProfile, summary: str,
        attempts: list[CodeAttempt], verified: bool,
    ) -> str:
        lines = [
            "# Local coding-agent analysis", "", f"**Requirement:** {request}",
            f"**Dataset:** {profile.filename} ({profile.row_count} rows)",
            f"**Columns:** {', '.join(profile.columns)}", f"**Summary:** {summary}",
            f"**Final verification:** {'PASSED' if verified else 'FAILED / UNVERIFIED'}", "",
            "## Execution attempts", "",
        ]
        for item in attempts:
            lines.extend([
                f"### Attempt {item.attempt} — {item.version}",
                f"- Model: {item.model_used}", f"- Executed: {item.executed}",
                f"- Exit code: {item.exit_code}", f"- Verified: {item.verified}",
                f"- stderr summary: `{item.stderr_summary[:500]}`", "",
            ])
        lines.append("Generated code was executed only in the restricted Docker sandbox; no host fallback is permitted.")
        return "\n".join(lines) + "\n"
