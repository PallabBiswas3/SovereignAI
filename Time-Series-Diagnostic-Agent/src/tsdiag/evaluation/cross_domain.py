from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..models import DiagnosticResult, json_safe
from ..pipeline import PIPELINE_VERSION


CROSS_DOMAIN_REPORT_VERSION = "1.0"
EXPECTED_DOMAINS = ("battery", "bearing", "process", "transformer", "turbofan", "wind_scada")


def _as_result_mapping(result: DiagnosticResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, DiagnosticResult):
        return result.to_dict()
    if not isinstance(result, Mapping):
        raise TypeError("cross-domain report inputs must be DiagnosticResult objects or mappings")
    return json_safe(dict(result))


def _run_record(result: DiagnosticResult | Mapping[str, Any]) -> dict[str, Any]:
    row = _as_result_mapping(result)
    metadata = dict(row.get("metadata") or {})
    provenance = dict(row.get("provenance") or {})
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return {
        "domain": str(row.get("domain", "unknown")),
        "task": str(row.get("task", "unknown")),
        "decision": str(row.get("decision", "unknown")),
        "abstained": bool(row.get("abstained", False)),
        "confidence": row.get("confidence"),
        "result_schema_version": row.get("schema_version"),
        "pipeline_version": metadata.get("pipeline_version", "unknown"),
        "workflow_version": provenance.get("workflow_version") or metadata.get("workflow_version"),
        "policy_version": provenance.get("policy_version") or metadata.get("policy_version"),
        "model_versions": provenance.get("model_versions") or metadata.get("resolved_model_versions") or {},
        "dataset_id": provenance.get("dataset_id"),
        "protocol_id": provenance.get("protocol_id"),
        "run_id": provenance.get("run_id"),
        "source": provenance.get("source"),
        "input_hash": provenance.get("input_hash"),
        "git_sha": provenance.get("git_sha"),
        "artifact_checksums": provenance.get("artifact_checksums") or {},
        "outcome": metadata.get("outcome") or {},
        "execution_summary": metadata.get("execution_summary") or {},
        "evidence_summary": metadata.get("evidence_summary") or {},
        "result_hash": sha256(canonical).hexdigest(),
    }


def build_cross_domain_report(
    results: Iterable[DiagnosticResult | Mapping[str, Any]],
    *,
    report_id: str | None = None,
    scope: str = "diagnostic-runs",
) -> dict[str, Any]:
    runs = [_run_record(result) for result in results]
    domains = sorted({row["domain"] for row in runs})
    decisions = Counter(row["decision"] for row in runs)
    versions = sorted({str(row["pipeline_version"]) for row in runs})
    return {
        "report_schema": "tsdiag.cross-domain-report",
        "report_version": CROSS_DOMAIN_REPORT_VERSION,
        "report_id": report_id,
        "scope": str(scope),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_pipeline_version": PIPELINE_VERSION,
        "summary": {
            "run_count": len(runs),
            "domains_present": domains,
            "domains_missing": sorted(set(EXPECTED_DOMAINS) - set(domains)),
            "decision_counts": dict(sorted(decisions.items())),
            "pipeline_versions": versions,
            "complete_six_domain_coverage": set(domains) == set(EXPECTED_DOMAINS),
        },
        "runs": runs,
    }


def write_cross_domain_report(
    report: Mapping[str, Any],
    output_dir: str | Path,
    *,
    stem: str = "cross_domain_report",
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = json_safe(dict(report))
    json_path = root / f"{stem}.json"
    markdown_path = root / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = dict(payload.get("summary") or {})
    lines = [
        "# Cross-domain diagnostic report",
        "",
        f"- Report schema version: `{payload.get('report_version')}`",
        f"- Generated: `{payload.get('generated_at_utc')}`",
        f"- Scope: `{payload.get('scope')}`",
        f"- Runs: `{summary.get('run_count', 0)}`",
        f"- Complete six-domain coverage: `{summary.get('complete_six_domain_coverage', False)}`",
        "",
        "| Domain | Task | Decision | Pipeline | Workflow | Policy | Dataset | Protocol | Git SHA |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload.get("runs", []):
        lines.append(
            "| " + " | ".join(str(row.get(key) or "") for key in (
                "domain", "task", "decision", "pipeline_version", "workflow_version",
                "policy_version", "dataset_id", "protocol_id", "git_sha",
            )) + " |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
