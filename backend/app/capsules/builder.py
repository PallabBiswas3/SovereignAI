from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from app.agent.state import AgentRunState
from app.artifacts.service import ArtifactService
from app.audit.logger import AuditLogger
from app.capsules.models import (
    CapsuleFileIdentity,
    CapsuleManifest,
    CapsuleSignature,
    CapsuleState,
    CapsuleWorkcellIdentity,
)
from app.capsules.signing import CapsuleSigner
from app.core.database import ArtifactRecord, AuditEventRecord, EvidenceCapsuleRecord, HumanApprovalRecord
from app.identity import ContentIdentityService
from app.workcells.models import WorkcellDefinition
from app.identity.models import Role


class EvidenceCapsuleBuilder:
    SCHEMA_VERSION = "1.0"

    def __init__(self, session: Session, root: Path, artifact_root: Path) -> None:
        self.session = session
        self.root = root.resolve()
        self.artifact_root = artifact_root.resolve()
        self.identity = ContentIdentityService()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")

    def _write_payload(self, root: Path, relative: str, value: Any) -> None:
        path = root / relative
        self._write_json(path, value)

    def build(
        self,
        task: AgentRunState,
        definition: WorkcellDefinition,
        *,
        inputs: list[dict[str, Any]] | None = None,
        signer: CapsuleSigner | None = None,
    ) -> EvidenceCapsuleRecord:
        if not task.workcell_id or task.workcell_id != definition.manifest.id:
            raise ValueError("Capsules require a completed Workcell task with matching identity")
        capsule_id = str(uuid4())
        record = EvidenceCapsuleRecord(
            id=capsule_id, task_id=task.id, state=CapsuleState.building.value,
            path=capsule_id, signature_status="UNSIGNED",
            organization_id=task.organization_id,
            owner_id=task.principal_id,
            workspace_id=task.workspace_id,
            department_id=task.department_id,
            classification=task.classification,
            allowed_roles_json=json.dumps([Role.approver.value, Role.manager.value, Role.auditor.value]),
        )
        self.session.add(record)
        self.session.commit()
        temporary = Path(tempfile.mkdtemp(prefix=f".{capsule_id}-", dir=self.root))
        destination = self.root / capsule_id
        try:
            resolved_inputs = list(inputs or [])
            if not resolved_inputs:
                seen_paths: set[str] = set()
                for item in task.evidence_records:
                    provenance = item.get("provenance", {}) if isinstance(item, dict) else {}
                    raw_path = provenance.get("path") if isinstance(provenance, dict) else None
                    if not raw_path or str(raw_path) in seen_paths:
                        continue
                    path = Path(str(raw_path)).resolve()
                    if path.is_file():
                        seen_paths.add(str(raw_path))
                        resolved_inputs.append({
                            "name": path.name, "sha256": self.identity.hash_file(path),
                            "size": path.stat().st_size, "reference_only": True,
                        })
            (temporary / "final_answer.md").write_text((task.final_response or "") + "\n", encoding="utf-8")
            self._write_payload(temporary, "inputs/manifest.json", {"request": task.request, "inputs": resolved_inputs})
            evidence = {
                "sources": task.sources,
                "fragments": task.evidence_records,
                "measurements": task.measurements,
                "rules": task.rules,
                "calculations": task.calculations,
                "claims": task.claims,
                "conflicts": task.conflicts,
            }
            for name, value in evidence.items():
                if value:
                    self._write_payload(temporary, f"evidence/{name}.json", value)
            # Operational evidence is a task-time snapshot. Verification later
            # hashes these stored values and never performs a fresh telemetry read.
            if task.asset_context:
                self._write_payload(temporary, "asset/asset_context_snapshot.json", task.asset_context)
            if task.trend_analyses:
                self._write_payload(temporary, "asset/trend_analyses.json", task.trend_analyses)
            if task.maintenance_history:
                self._write_payload(temporary, "asset/maintenance_history.json", task.maintenance_history)
            if task.maintenance_draft:
                self._write_payload(temporary, "asset/maintenance_draft.json", task.maintenance_draft)
            self._write_payload(temporary, "execution/workcell_manifest.json", definition.manifest.model_dump(mode="json"))
            self._write_payload(temporary, "execution/workflow_definition.json", definition.workflow.model_dump(mode="json"))
            self._write_payload(temporary, "execution/workcell_file_manifest.json", definition.files)
            self._write_payload(temporary, "execution/prompt_manifest.json", [
                {"path": name, "sha256": digest} for name, digest in sorted(definition.prompt_hashes.items())
            ])
            self._write_payload(temporary, "execution/model_manifest.json", [{
                "role": task.routing.model_id.upper(),
                "provider": "ollama",
                "model": task.routing.selected_model,
                "digest": None,
                "digest_verified": False,
                "runtime_parameters": {"execution_mode": task.execution_mode},
            }])
            self._write_payload(temporary, "execution/policy_decisions.json", task.governance)
            self._write_payload(temporary, "execution/tool_calls.json", task.tool_records)
            approvals = self.session.query(HumanApprovalRecord).filter(HumanApprovalRecord.run_id == task.id).all()
            self._write_payload(temporary, "execution/human_decisions.json", [{
                "id": row.id, "tool": row.tool, "status": row.status,
                "decided_by": row.decided_by, "decided_at": row.decided_at,
            } for row in approvals])
            audits = self.session.query(AuditEventRecord).filter(AuditEventRecord.run_id == task.id).order_by(AuditEventRecord.created_at).all()
            audit_path = temporary / "execution" / "audit.jsonl"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text("".join(json.dumps({
                "event_type": row.event_type, "summary": row.summary,
                "payload": json.loads(row.payload_json), "created_at": row.created_at.isoformat(),
            }, ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in audits), encoding="utf-8")

            artifact_rows = self.session.query(ArtifactRecord).filter(ArtifactRecord.run_id == task.id).all()
            artifact_service = ArtifactService(self.session, self.artifact_root)
            artifact_manifest: list[dict[str, Any]] = []
            for row in sorted(artifact_rows, key=lambda item: item.name):
                source = artifact_service.resolve(row)
                if not source.is_file():
                    continue
                safe_name = Path(row.name).name
                target = temporary / "artifacts" / safe_name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                artifact_manifest.append({
                    "artifact_id": row.id, "task_id": task.id,
                    "workcell_id": task.workcell_id, "workcell_version": task.workcell_version,
                    "artifact_type": row.artifact_type or source.suffix.lstrip("."),
                    "path": f"artifacts/{safe_name}", "sha256": self.identity.hash_file(target),
                    "created_at": row.created_at.isoformat(),
                    "lineage": json.loads(row.lineage_json or "{}"),
                })

            payload_paths = sorted(
                path for path in temporary.rglob("*")
                if path.is_file() and path.name not in {"capsule_manifest.json", "hashes.sha256", "signature.json"}
            )
            files = [CapsuleFileIdentity(
                path=path.relative_to(temporary).as_posix(),
                sha256=self.identity.hash_file(path), size=path.stat().st_size,
                category=path.relative_to(temporary).parts[0],
            ) for path in payload_paths]
            hash_map = {item.path: item.sha256 for item in files}
            root_hash = self.identity.hash_directory_manifest(hash_map)
            manifest = CapsuleManifest(
                capsule_id=capsule_id, task_id=task.id,
                request_summary=task.request[:2_000],
                workcell=CapsuleWorkcellIdentity(
                    id=task.workcell_id, version=task.workcell_version or "unknown",
                    hash=task.workcell_hash or definition.content_hash,
                    workflow_version=task.workflow_version or definition.workflow.version,
                ),
                models=json.loads((temporary / "execution/model_manifest.json").read_text(encoding="utf-8")),
                inputs=resolved_inputs, artifacts=artifact_manifest, files=files,
                capsule_root_hash=root_hash,
            )
            self._write_json(temporary / "capsule_manifest.json", manifest.model_dump(mode="json"))
            (temporary / "hashes.sha256").write_text(
                "".join(f"{item.sha256}  {item.path}\n" for item in sorted(files, key=lambda value: value.path)),
                encoding="utf-8",
            )
            signature: CapsuleSignature | None = signer.sign(root_hash) if signer else None
            if signature:
                self._write_json(temporary / "signature.json", signature.model_dump(mode="json"))
                record.signature_status = "SIGNED_UNVERIFIED"
            temporary.rename(destination)
            record.state = CapsuleState.complete.value
            record.path = destination.name
            record.capsule_root_hash = root_hash
            record.manifest_json = manifest.model_dump_json()
            record.updated_at = datetime.now(timezone.utc)
            self.session.commit()
            AuditLogger(self.session).log(task.id, "capsule_created", f"Evidence Capsule {capsule_id} created", {
                "capsule_id": capsule_id, "capsule_root_hash": root_hash,
                "file_count": len(files), "signature_status": record.signature_status,
            })
            AuditLogger(self.session).log(task.id, "capsule_hash", "Capsule root hash calculated from the canonical sorted payload manifest.", {
                "capsule_id": capsule_id, "capsule_root_hash": root_hash,
                "algorithm": manifest.root_hash_algorithm,
            })
            if signature:
                AuditLogger(self.session).log(task.id, "capsule_signing", f"Capsule signed with {signature.algorithm} key {signature.key_id}.", {
                    "capsule_id": capsule_id, "algorithm": signature.algorithm,
                    "key_id": signature.key_id, "signed_root_hash": signature.signed_root_hash,
                })
            return record
        except Exception:
            record.state = CapsuleState.failed.value
            record.updated_at = datetime.now(timezone.utc)
            self.session.commit()
            shutil.rmtree(temporary, ignore_errors=True)
            raise
