from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent.state import AgentPlan, AgentRunState, AgentStep, RunStatus
from app.artifacts.service import ArtifactService
from app.capsules.builder import EvidenceCapsuleBuilder
from app.capsules.models import SignatureStatus
from app.capsules.signing import Ed25519CapsuleSigner, WorkcellTrustStore
from app.capsules.verifier import EvidenceCapsuleVerifier
from app.core.database import Base
from app.main import app
from app.router.schemas import RoutingDecision, TaskProfile
from app.workcells.loader import WorkcellLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def session_for(tmp_path: Path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def task_state(task_id: str = "task-capsule") -> AgentRunState:
    routing = RoutingDecision(
        selected_model="qwen3-vl:4b", model_id="general", confidence=0.9,
        reason="inspection task", scores={"general": 1.0},
        task_profile=TaskProfile(
            task_type="engineering_inspection", coding_requirement=0.0,
            reasoning_requirement=0.8, vision_requirement=0.2,
            document_requirement=0.9, summarization_requirement=0.5,
            latency_priority=0.2, context_length_required=4096,
        ),
    )
    return AgentRunState(
        id=task_id, request="Assess Pump-102", status=RunStatus.completed,
        plan=AgentPlan(goal="Assess Pump-102", steps=[AgentStep(id=1, action="verify", title="Verify")]),
        routing=routing, final_response="Pump-102 requires planned maintenance.",
        workcell_id="pump-inspection", workcell_version="1.0.0",
        workcell_hash="0" * 64, workflow_version="1.0.0",
        sources=[{"id": "SRC1", "file": "SOP.md", "section": "4.2", "revision": "R1", "document_hash": "a" * 64}],
        measurements=[{"id": "M1", "metric": "vibration", "original_value": 6.1, "original_unit": "mm/s", "source_id": "inspection"}],
        rules=[{"id": "R1", "metric": "vibration", "operator": "<=", "threshold": 4.5, "unit": "mm/s", "rule_type": "normal_limit", "source": {"source_id": "SRC1"}}],
        calculations=[{"id": "C1", "expression": "6.1 <= 4.5", "inputs": ["M1", "R1"], "result": False, "verified": True}],
        claims=[{"id": "CL1", "text": "Vibration exceeds the normal limit.", "claim_type": "engineering_finding", "evidence_ids": ["M1", "R1"], "calculation_ids": ["C1"], "support_status": "SUPPORTED"}],
        governance={"decision": "ALLOW", "policy": "engineering"},
    )


def build_fixture(tmp_path: Path, signer=None):
    db = session_for(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_path = artifact_root / "task-capsule" / "approval_note.docx"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"test-docx-content")
    ArtifactService(db, artifact_root).register(
        artifact_path, "task-capsule", workcell_id="pump-inspection",
        workcell_version="1.0.0", artifact_type="docx", derived_from_claims=["CL1"],
    )
    definition = WorkcellLoader(PROJECT_ROOT / "workcells").load("pump_inspection")
    task = task_state()
    task.workcell_hash = definition.content_hash
    record = EvidenceCapsuleBuilder(db, tmp_path / "capsules", artifact_root).build(
        task, definition, inputs=[{"name": "inspection.md", "sha256": "b" * 64}], signer=signer,
    )
    return db, record, tmp_path / "capsules" / record.path


def test_workcell_catalog_api_exposes_ready_versioned_pack():
    with TestClient(app) as client:
        response = client.get("/api/workcells")
        assert response.status_code == 200
        pump = next(item for item in response.json() if item["id"] == "pump-inspection")
        assert pump["version"] == "1.0.0"
        assert pump["status"] == "READY"
        detail = client.post("/api/workcells/pump-inspection/validate")
        assert detail.status_code == 200
        assert detail.json()["valid"] is True


def test_capsule_creation_manifest_hashes_root_and_unsigned_policies(tmp_path: Path):
    _, record, root = build_fixture(tmp_path)
    assert record.state == "COMPLETE"
    assert (root / "capsule_manifest.json").is_file()
    assert (root / "hashes.sha256").is_file()
    assert (root / "evidence" / "claims.json").is_file()
    manifest = json.loads((root / "capsule_manifest.json").read_text(encoding="utf-8"))
    assert manifest["capsule_root_hash"] == record.capsule_root_hash
    assert manifest["workcell"]["id"] == "pump-inspection"
    assert manifest["artifacts"][0]["lineage"]["derived_from_claims"] == ["CL1"]
    development = EvidenceCapsuleVerifier(unsigned_allowed=True).verify(root)
    assert development.status.value == "VALID"
    assert development.signature_status == SignatureStatus.unsigned
    strict = EvidenceCapsuleVerifier(unsigned_allowed=False).verify(root)
    assert strict.status.value == "INVALID"
    assert strict.signature_status == SignatureStatus.unsigned


def test_capsule_tamper_identifies_exact_file(tmp_path: Path):
    _, _, root = build_fixture(tmp_path)
    target = root / "artifacts" / "approval_note.docx"
    target.write_bytes(target.read_bytes() + b"!")
    result = EvidenceCapsuleVerifier().verify(root)
    assert result.status.value == "INVALID"
    assert any(item.path == "artifacts/approval_note.docx" and item.type == "CAPSULE_HASH_MISMATCH" for item in result.failures)


def test_capsule_missing_extra_and_invalid_manifest_are_rejected(tmp_path: Path):
    _, _, root = build_fixture(tmp_path)
    (root / "evidence" / "claims.json").unlink()
    (root / "evidence" / "extra.json").write_text("{}", encoding="utf-8")
    result = EvidenceCapsuleVerifier().verify(root)
    types = {item.type for item in result.failures}
    assert {"MISSING_FILE", "UNEXPECTED_FILE"} <= types
    (root / "capsule_manifest.json").write_text("{}", encoding="utf-8")
    invalid = EvidenceCapsuleVerifier().verify(root)
    assert invalid.status.value == "INVALID"
    assert invalid.failures[0].type == "CAPSULE_SCHEMA_INVALID"


def test_capsule_artifact_and_workcell_metadata_inconsistency_is_rejected(tmp_path: Path):
    _, _, root = build_fixture(tmp_path)
    manifest_path = root / "capsule_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["sha256"] = "f" * 64
    manifest["workcell"]["hash"] = "e" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = EvidenceCapsuleVerifier().verify(root)
    types = {item.type for item in result.failures}
    assert {"ARTIFACT_IDENTITY_MISMATCH", "WORKCELL_IDENTITY_MISMATCH"} <= types


def test_ed25519_signature_valid_wrong_key_and_tamper(tmp_path: Path):
    signer = Ed25519CapsuleSigner.generate_for_testing("batch3-test")
    _, _, root = build_fixture(tmp_path, signer=signer)
    trust = WorkcellTrustStore()
    trust.add_ed25519("batch3-test", signer.public_bytes())
    valid = EvidenceCapsuleVerifier(trust).verify(root)
    assert valid.status.value == "VALID"
    assert valid.signature_status == SignatureStatus.valid
    wrong = Ed25519CapsuleSigner.generate_for_testing("batch3-test")
    wrong_trust = WorkcellTrustStore()
    wrong_trust.add_ed25519("batch3-test", wrong.public_bytes())
    rejected = EvidenceCapsuleVerifier(wrong_trust).verify(root)
    assert rejected.status.value == "INVALID"
    assert rejected.signature_status == SignatureStatus.invalid
    (root / "final_answer.md").write_text("changed", encoding="utf-8")
    tampered = EvidenceCapsuleVerifier(trust).verify(root)
    assert tampered.status.value == "INVALID"
    assert any(item.path == "final_answer.md" for item in tampered.failures)
