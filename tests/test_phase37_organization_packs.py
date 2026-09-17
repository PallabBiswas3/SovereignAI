from __future__ import annotations

from pathlib import Path
import shutil
import asyncio

import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.core.database import (
    AssetEvidenceLinkRecord, AssetRecord, AuditEventRecord, Base,
    KnowledgeDocument, OrganizationRecord, UserRecord,
)
from app.api.organization import (
    OrganizationPackImportRequest, import_organization_pack, validate_organization_pack,
)
from app.identity.models import ClearanceLevel, Permission, Principal, Role
from app.identity.passwords import PasswordHasher
from app.organizations.importer import OrganizationImportError, OrganizationPackImporter
from app.organizations.loader import OrganizationPackError, OrganizationPackLoader
from app.rag.embeddings import LocalHashEmbeddingProvider
from app.workcells.defaults import configured_workcell_registry


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def create_pack(root: Path) -> Path:
    _write_yaml(root / "organization.yaml", {
        "schema_version": "1.0",
        "organization": {"id": "nova", "name": "Nova Industrial Ltd.", "short_name": "NOVA"},
        "plants": [{"id": "nova-plant-1", "name": "Plant 1"}],
        "areas": [{"id": "nova-utilities", "name": "Utilities", "plant_id": "nova-plant-1"}],
        "units": [{"id": "nova-water", "name": "Cooling Water", "plant_id": "nova-plant-1", "area_id": "nova-utilities"}],
        "departments": [{"id": "nova-maintenance", "name": "Maintenance"}],
        "workspaces": [{"id": "nova-plant-1-workspace", "name": "Plant 1"}],
    })
    _write_yaml(root / "access_matrix.yaml", {
        "schema_version": "1.0",
        "asset_rules": [{
            "id": "nova-maint-assets", "workspace_id": "nova-plant-1-workspace",
            "department_id": "nova-maintenance", "classification": "confidential",
            "allowed_roles": ["ENGINEER"],
        }],
        "document_rules": [{
            "id": "nova-maint-docs", "workspace_id": "nova-plant-1-workspace",
            "department_id": "nova-maintenance", "classification": "confidential",
            "allowed_roles": ["ENGINEER"],
        }],
    })
    _write_yaml(root / "assets.yaml", {
        "schema_version": "1.0",
        "assets": [{
            "id": "NOVA-Pump-001", "canonical_name": "Cooling Water Pump",
            "asset_type": "pump", "plant_id": "nova-plant-1", "area_id": "nova-utilities",
            "unit_id": "nova-water", "access_rule_id": "nova-maint-assets",
            "criticality": "HIGH", "status": "IN_SERVICE", "aliases": ["NOVA-P-001"],
        }],
    })
    _write_yaml(root / "users.yaml", {
        "schema_version": "1.0", "identity_provider": "local",
        "users": [{
            "id": "nova-engineer-1", "email": "engineer@nova.example",
            "display_name": "Nova Engineer", "departments": ["nova-maintenance"],
            "workspaces": ["nova-plant-1-workspace"], "roles": ["USER", "ENGINEER"],
            "clearance": "confidential", "password_env": "NOVA_ENGINEER_PASSWORD",
        }],
    })
    _write_yaml(root / "documents/manifest.yaml", {
        "schema_version": "1.0",
        "documents": [{
            "path": "maintenance/pump-001-sop.md", "access_rule_id": "nova-maint-docs",
            "revision": "Rev 1", "approval_status": "APPROVED",
            "asset_ids": ["NOVA-Pump-001"], "relationship": "HAS_REQUIREMENT",
        }],
    })
    document = root / "documents/maintenance/pump-001-sop.md"
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text("# NOVA Pump SOP\nNormal vibration shall not exceed 6.0 mm/s RMS.\n", encoding="utf-8")
    return root


def _session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return Session(engine)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        workcells_root=tmp_path / "installed-workcells",
        organizations_root=tmp_path / "organizations",
        organization_reports_root=tmp_path / "reports",
    )


def _admin(organization_id: str = "nova") -> Principal:
    return Principal(
        user_id=f"{organization_id}-admin", email=f"admin@{organization_id}.example",
        display_name="Organization Admin", organization_id=organization_id,
        department_ids=[f"{organization_id}-maintenance"],
        workspace_ids=[f"{organization_id}-plant-1-workspace"],
        roles=[Role.admin], permissions=[Permission.admin_manage_users, Permission.workcell_manage],
        clearance=ClearanceLevel.restricted,
    )


def test_organization_pack_loader_validates_complete_pack(tmp_path: Path) -> None:
    pack = OrganizationPackLoader(create_pack(tmp_path / "nova")).load()
    validation = OrganizationPackLoader(Path(pack.root)).validate(pack)
    assert validation.valid is True
    assert validation.organization_id == "nova"
    assert validation.counts["assets"] == 1
    assert validation.counts["documents"] == 1
    assert validation.required_secret_environment_variables == ["NOVA_ENGINEER_PASSWORD"]
    assert len(pack.pack_fingerprint) == 64


def test_organization_pack_rejects_plaintext_password(tmp_path: Path) -> None:
    root = create_pack(tmp_path / "nova")
    users = yaml.safe_load((root / "users.yaml").read_text(encoding="utf-8"))
    users["users"][0]["password"] = "must-never-be-stored-here"
    _write_yaml(root / "users.yaml", users)
    with pytest.raises(OrganizationPackError, match="extra_forbidden"):
        OrganizationPackLoader(root).load()


def test_organization_pack_rejects_ambiguous_asset_aliases(tmp_path: Path) -> None:
    root = create_pack(tmp_path / "nova")
    assets = yaml.safe_load((root / "assets.yaml").read_text(encoding="utf-8"))
    assets["assets"].append({
        **assets["assets"][0], "id": "NOVA-Pump-002", "aliases": ["NOVA-P-001"],
    })
    _write_yaml(root / "assets.yaml", assets)
    with pytest.raises(OrganizationPackError, match="AMBIGUOUS_ASSET_ALIAS"):
        OrganizationPackLoader(root).load()


def test_organization_pack_rejects_document_path_traversal(tmp_path: Path) -> None:
    root = create_pack(tmp_path / "nova")
    manifest = yaml.safe_load((root / "documents/manifest.yaml").read_text(encoding="utf-8"))
    manifest["documents"][0]["path"] = "../outside.md"
    _write_yaml(root / "documents/manifest.yaml", manifest)
    with pytest.raises(OrganizationPackError, match="safe relative path"):
        OrganizationPackLoader(root).load()


def test_importer_dry_run_does_not_mutate_database_or_require_secret(tmp_path: Path) -> None:
    pack = OrganizationPackLoader(create_pack(tmp_path / "nova")).load()
    with _session() as session:
        report = OrganizationPackImporter(
            session, settings=_settings(tmp_path), embeddings=LocalHashEmbeddingProvider(),
            secret_resolver=lambda _name: None,
        ).import_pack(pack, dry_run=True)
        assert report.status == "DRY_RUN"
        assert "NOVA_ENGINEER_PASSWORD" in report.required_secret_environment_variables
        assert session.query(OrganizationRecord).count() == 0
        assert Path(report.report_path or "").is_file()


def test_importer_is_idempotent_and_preserves_acl_asset_lineage(tmp_path: Path) -> None:
    pack = OrganizationPackLoader(create_pack(tmp_path / "nova")).load()
    settings = _settings(tmp_path)
    with _session() as session:
        importer = OrganizationPackImporter(
            session, settings=settings, embeddings=LocalHashEmbeddingProvider(),
            secret_resolver=lambda name: "NovaDemo!2026" if name == "NOVA_ENGINEER_PASSWORD" else None,
        )
        first = importer.import_pack(pack)
        second = importer.import_pack(pack)

        assert first.status == second.status == "IMPORTED"
        assert session.query(OrganizationRecord).count() == 1
        assert session.query(UserRecord).count() == 1
        assert session.query(AssetRecord).count() == 1
        assert session.query(KnowledgeDocument).count() == 1
        assert session.query(AssetEvidenceLinkRecord).count() == 1
        assert session.query(AuditEventRecord).count() == 2
        user = session.get(UserRecord, "nova-engineer-1")
        assert user and PasswordHasher().verify("NovaDemo!2026", user.password_hash)
        document = session.query(KnowledgeDocument).one()
        assert document.organization_id == "nova"
        assert document.department_id == "nova-maintenance"
        assert document.workspace_id == "nova-plant-1-workspace"
        assert document.classification == "CONFIDENTIAL"
        assert "NovaDemo!2026" not in Path(first.report_path or "").read_text(encoding="utf-8")


def test_importer_fails_before_mutation_when_password_secret_is_missing(tmp_path: Path) -> None:
    pack = OrganizationPackLoader(create_pack(tmp_path / "nova")).load()
    with _session() as session:
        importer = OrganizationPackImporter(
            session, settings=_settings(tmp_path), embeddings=LocalHashEmbeddingProvider(),
            secret_resolver=lambda _name: None,
        )
        with pytest.raises(OrganizationImportError, match="Missing required secret"):
            importer.import_pack(pack)
        assert session.query(OrganizationRecord).count() == 0


def test_importer_rejects_cross_organization_asset_id_collision(tmp_path: Path) -> None:
    pack = OrganizationPackLoader(create_pack(tmp_path / "nova")).load()
    with _session() as session:
        session.add(OrganizationRecord(id="other", name="Other", short_name="OTHER"))
        session.add(AssetRecord(
            id="NOVA-Pump-001", canonical_name="Other Pump", asset_type="pump",
            organization_id="other", plant_id="other-plant", area_id="other-area",
            unit_id="other-unit", workspace_id="other-workspace", criticality="LOW",
            status="IN_SERVICE",
        ))
        session.commit()
        importer = OrganizationPackImporter(
            session, settings=_settings(tmp_path), embeddings=LocalHashEmbeddingProvider(),
            secret_resolver=lambda _name: "NovaDemo!2026",
        )
        with pytest.raises(OrganizationImportError, match="CROSS_ORGANIZATION_ID_COLLISION"):
            importer.import_pack(pack)
        assert session.get(AssetRecord, "NOVA-Pump-001").organization_id == "other"


def test_imported_workcell_is_visible_only_inside_its_organization(tmp_path: Path) -> None:
    root = create_pack(tmp_path / "nova")
    source = Path(__file__).resolve().parents[1] / "workcells" / "pump_inspection"
    target = root / "workcells" / "nova_pump_inspection"
    shutil.copytree(source, target)
    manifest_path = target / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["id"] = "nova-pump-inspection"
    manifest["name"] = "NOVA Pump Inspection"
    manifest["organization_id"] = "nova"
    _write_yaml(manifest_path, manifest)
    pack = OrganizationPackLoader(root).load()
    settings = _settings(tmp_path)

    with _session() as session:
        report = OrganizationPackImporter(
            session, settings=settings, embeddings=LocalHashEmbeddingProvider(),
            secret_resolver=lambda _name: "NovaDemo!2026",
        ).import_pack(pack)

    assert report.workcells[0].valid is True
    registry = configured_workcell_registry(settings)
    assert any(item.id == "nova-pump-inspection" for item in registry.list("nova"))
    assert all(item.id != "nova-pump-inspection" for item in registry.list("other"))
    with pytest.raises(KeyError, match="WORKCELL_NOT_FOUND"):
        registry.get("nova-pump-inspection", organization_id="other")


def test_admin_api_validates_and_imports_only_own_organization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    organizations = tmp_path / "organizations"
    create_pack(organizations / "nova")
    settings = Settings(
        organizations_root=organizations, workcells_root=tmp_path / "workcells",
        organization_reports_root=tmp_path / "reports",
        embedding_provider="hash",
    )
    monkeypatch.setattr("app.api.organization.get_settings", lambda: settings)
    monkeypatch.setenv("NOVA_ENGINEER_PASSWORD", "NovaDemo!2026")
    with _session() as session:
        validation = asyncio.run(validate_organization_pack("nova", _admin(), session))
        assert validation["valid"] is True
        report = asyncio.run(import_organization_pack(
            "nova",
            OrganizationPackImportRequest(
                confirm_organization_id="nova", dry_run=False,
            ),
            _admin(), session,
        ))
        assert report["status"] == "IMPORTED"
        audit = session.query(AuditEventRecord).one()
        assert audit.principal_id == "nova-admin"
        with pytest.raises(Exception) as denied:
            asyncio.run(validate_organization_pack("nova", _admin("other"), session))
        assert getattr(denied.value, "status_code", None) == 403
