from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.database import (
    AreaRecord, AssetAliasRecord, AssetEvidenceLinkRecord, AssetRecord, AuditEventRecord,
    DepartmentRecord, KnowledgeDocument, OrganizationRecord, PlantRecord, UnitRecord,
    UserRecord, WorkspaceRecord,
)
from app.identity.models import DocumentACL
from app.identity.models import Principal
from app.identity.passwords import PasswordHasher
from app.organizations.loader import OrganizationPackError, OrganizationPackLoader
from app.organizations.models import (
    AccessRule, LoadedOrganizationPack, OrganizationImportReport,
    OrganizationPackValidation, PackValidationIssue, WorkcellPackStatus,
)
from app.rag.embeddings import EmbeddingProvider, configured_embedding_provider
from app.rag.ingestion import KnowledgeIngestionService
from app.workcells.defaults import create_workcell_handler_registry
from app.workcells.loader import WorkcellLoader
from app.workcells.validator import WorkcellValidator


SecretResolver = Callable[[str], str | None]


class OrganizationImportError(ValueError):
    pass


class OrganizationPackImporter:
    """Fail-closed, idempotent importer for a previously loaded Organization Pack."""

    def __init__(
        self,
        session: Session,
        *,
        settings: Settings | None = None,
        embeddings: EmbeddingProvider | None = None,
        secret_resolver: SecretResolver | None = None,
        principal: Principal | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.embeddings = embeddings
        self.secret_resolver = secret_resolver or os.environ.get
        self.principal = principal
        self.password_hasher = PasswordHasher()

    def validate(self, pack: LoadedOrganizationPack) -> OrganizationPackValidation:
        validation = OrganizationPackLoader(Path(pack.root)).validate(pack)
        issues = list(validation.issues)
        issues.extend(self._database_collision_issues(pack))
        workcells = self._validate_workcells(pack, issues)
        return validation.model_copy(update={
            "valid": not issues,
            "issues": issues,
            "workcells": workcells,
        })

    def import_pack(
        self, pack: LoadedOrganizationPack, *, dry_run: bool = False,
        rotate_local_passwords: bool = False,
    ) -> OrganizationImportReport:
        validation = self.validate(pack)
        if not validation.valid:
            raise OrganizationImportError(json.dumps(validation.model_dump(mode="json"), indent=2))
        missing_secrets = [
            name for name in validation.required_secret_environment_variables
            if not self.secret_resolver(name)
        ]
        warnings = [
            "Imports are additive and idempotent; records omitted from a later pack are not deleted."
        ]
        if pack.users_file.identity_provider == "external":
            warnings.append(
                "External identity definitions are staged disabled until an external IdentityProvider adapter is configured."
            )
        if missing_secrets:
            warnings.append(f"Missing local password environment variables: {missing_secrets}")
        report = OrganizationImportReport(
            status="DRY_RUN" if dry_run else "IMPORTED",
            organization_id=pack.organization_id,
            pack_fingerprint=pack.pack_fingerprint,
            counts=validation.counts,
            workcells=validation.workcells,
            required_secret_environment_variables=validation.required_secret_environment_variables,
            warnings=warnings,
        )
        if dry_run:
            return self._write_report(report)
        if missing_secrets:
            raise OrganizationImportError(
                f"Missing required secret environment variables: {missing_secrets}"
            )

        installed: list[Path] = []
        try:
            installed = self._install_workcells(pack, validation.workcells)
            self._upsert_structure(pack)
            self._upsert_users(pack, rotate_local_passwords=rotate_local_passwords)
            self._upsert_assets(pack)
            self._ingest_documents(pack)
            self.session.add(AuditEventRecord(
                id=str(uuid4()), run_id=f"organization-import:{pack.pack_fingerprint[:16]}",
                event_type="ORGANIZATION_PACK_IMPORTED",
                summary=f"Imported validated Organization Pack for {pack.organization_id}",
                payload_json=json.dumps({
                    "organization_id": pack.organization_id,
                    "pack_fingerprint": pack.pack_fingerprint,
                    "counts": validation.counts,
                }),
                principal_id=self.principal.user_id if self.principal else None,
                organization_id=pack.organization_id,
            ))
            self.session.commit()
        except Exception as exc:
            self.session.rollback()
            for path in installed:
                resolved = path.resolve()
                root = self.settings.workcells_root.resolve()
                if root in resolved.parents and resolved.is_dir():
                    shutil.rmtree(resolved)
            if isinstance(exc, OrganizationImportError):
                raise
            raise OrganizationImportError(str(exc)) from exc
        return self._write_report(report)

    def _database_collision_issues(self, pack: LoadedOrganizationPack) -> list[PackValidationIssue]:
        issues: list[PackValidationIssue] = []
        organization_id = pack.organization_id

        def collision(model, identifier: str, path: str) -> None:
            row = self.session.get(model, identifier)
            if row and getattr(row, "organization_id", organization_id) != organization_id:
                issues.append(PackValidationIssue(
                    code="CROSS_ORGANIZATION_ID_COLLISION",
                    message=f"{model.__name__} ID {identifier} already belongs to another organization",
                    path=path,
                ))

        for item in pack.organization_file.plants:
            collision(PlantRecord, item.id, "organization.yaml")
        for item in pack.organization_file.areas:
            collision(AreaRecord, item.id, "organization.yaml")
        for item in pack.organization_file.units:
            collision(UnitRecord, item.id, "organization.yaml")
        for item in pack.organization_file.departments:
            collision(DepartmentRecord, item.id, "organization.yaml")
        for item in pack.organization_file.workspaces:
            collision(WorkspaceRecord, item.id, "organization.yaml")
        for item in pack.assets_file.assets:
            collision(AssetRecord, item.id, "assets.yaml")
        for user in pack.users_file.users:
            collision(UserRecord, user.id, "users.yaml")
            email_owner = self.session.query(UserRecord).filter(
                UserRecord.email_normalized == user.email.lower()
            ).first()
            if email_owner and email_owner.organization_id != organization_id:
                issues.append(PackValidationIssue(
                    code="CROSS_ORGANIZATION_EMAIL_COLLISION",
                    message=f"Email {user.email} already belongs to another organization",
                    path="users.yaml",
                ))
        root = Path(pack.root) / "documents"
        for document in pack.documents_file.documents:
            checksum = hashlib.sha256((root / document.path).read_bytes()).hexdigest()
            existing = self.session.query(KnowledgeDocument).filter_by(checksum=checksum).first()
            if existing and existing.organization_id not in {None, organization_id}:
                issues.append(PackValidationIssue(
                    code="CROSS_ORGANIZATION_DOCUMENT_COLLISION",
                    message=f"Document {document.path} has identical content already scoped to another organization",
                    path=document.path,
                ))
        return issues

    def _validate_workcells(
        self, pack: LoadedOrganizationPack, issues: list[PackValidationIssue],
    ) -> list[WorkcellPackStatus]:
        statuses: list[WorkcellPackStatus] = []
        if not pack.workcell_directories:
            return statuses
        loader = WorkcellLoader(Path(pack.root) / "workcells")
        validator = WorkcellValidator(
            create_workcell_handler_registry(), self.settings.tools_config,
            unsigned_workcells_allowed=self.settings.unsigned_workcells_allowed,
        )
        existing: dict[tuple[str, str], str | None] = {}
        pack_keys: set[tuple[str, str]] = set()
        if self.settings.workcells_root.exists():
            global_loader = WorkcellLoader(self.settings.workcells_root)
            for directory in self.settings.workcells_root.iterdir():
                if not directory.is_dir():
                    continue
                try:
                    definition = global_loader.load(directory)
                    existing[(definition.manifest.id, definition.manifest.version)] = definition.content_hash
                except Exception:
                    continue
        for raw_directory in pack.workcell_directories:
            directory = Path(raw_directory)
            try:
                definition = loader.load(directory)
                result = validator.validate(definition)
                messages = [item.message for item in result.issues]
                key = (definition.manifest.id, definition.manifest.version)
                if key in pack_keys:
                    result.valid = False
                    messages.append("Duplicate Workcell ID and version inside Organization Pack")
                pack_keys.add(key)
                if definition.manifest.organization_id != pack.organization_id:
                    result.valid = False
                    messages.append(
                        f"Workcell organization_id must equal {pack.organization_id}"
                    )
                existing_hash = existing.get((definition.manifest.id, definition.manifest.version))
                if existing_hash and existing_hash != definition.content_hash:
                    result.valid = False
                    messages.append("A different Workcell with the same ID and version is installed")
                status = WorkcellPackStatus(
                    directory=directory.name, workcell_id=definition.manifest.id,
                    version=definition.manifest.version, content_hash=definition.content_hash,
                    valid=result.valid,
                    status=result.status.value if result.valid else "INVALID",
                    issues=messages,
                )
            except Exception as exc:
                status = WorkcellPackStatus(
                    directory=directory.name, valid=False, status="INVALID", issues=[str(exc)],
                )
            statuses.append(status)
            if not status.valid:
                issues.append(PackValidationIssue(
                    code="ORGANIZATION_WORKCELL_INVALID",
                    message="; ".join(status.issues) or f"Invalid Workcell {status.directory}",
                    path=f"workcells/{status.directory}",
                ))
        return statuses

    def _upsert_structure(self, pack: LoadedOrganizationPack) -> None:
        source = pack.organization_file
        spec = source.organization
        row = self.session.get(OrganizationRecord, spec.id) or OrganizationRecord(
            id=spec.id, name=spec.name, short_name=spec.short_name,
        )
        row.name, row.short_name = spec.name, spec.short_name
        row.metadata_json = json.dumps({
            **spec.metadata, "organization_pack": {
                "schema_version": source.schema_version,
                "fingerprint": pack.pack_fingerprint,
            },
        }, ensure_ascii=False)
        self.session.add(row)
        for item in source.departments:
            record = self.session.get(DepartmentRecord, item.id) or DepartmentRecord(
                id=item.id, organization_id=pack.organization_id, name=item.name,
            )
            record.organization_id, record.name = pack.organization_id, item.name
            self.session.add(record)
        for item in source.workspaces:
            record = self.session.get(WorkspaceRecord, item.id) or WorkspaceRecord(
                id=item.id, organization_id=pack.organization_id, name=item.name,
            )
            record.organization_id, record.name = pack.organization_id, item.name
            self.session.add(record)
        for item in source.plants:
            record = self.session.get(PlantRecord, item.id) or PlantRecord(
                id=item.id, organization_id=pack.organization_id, name=item.name,
            )
            record.organization_id, record.name = pack.organization_id, item.name
            self.session.add(record)
        for item in source.areas:
            record = self.session.get(AreaRecord, item.id) or AreaRecord(
                id=item.id, organization_id=pack.organization_id,
                plant_id=item.plant_id, name=item.name,
            )
            record.organization_id, record.plant_id, record.name = pack.organization_id, item.plant_id, item.name
            self.session.add(record)
        for item in source.units:
            record = self.session.get(UnitRecord, item.id) or UnitRecord(
                id=item.id, organization_id=pack.organization_id, plant_id=item.plant_id,
                area_id=item.area_id, name=item.name,
            )
            record.organization_id, record.plant_id = pack.organization_id, item.plant_id
            record.area_id, record.name = item.area_id, item.name
            self.session.add(record)
        self.session.flush()

    def _upsert_users(self, pack: LoadedOrganizationPack, *, rotate_local_passwords: bool) -> None:
        local = pack.users_file.identity_provider == "local"
        for user in pack.users_file.users:
            record = self.session.get(UserRecord, user.id)
            if record is None:
                password = self.secret_resolver(user.password_env or "") if local else None
                password_hash = self.password_hasher.hash(password or "external-identity-disabled")
                record = UserRecord(
                    id=user.id, email=user.email, email_normalized=user.email.lower(),
                    display_name=user.display_name, organization_id=pack.organization_id,
                    password_hash=password_hash,
                )
            elif local and rotate_local_passwords:
                password = self.secret_resolver(user.password_env or "")
                if not password:
                    raise OrganizationImportError(f"Password secret is unavailable for {user.id}")
                record.password_hash = self.password_hasher.hash(password)
            record.email, record.email_normalized = user.email, user.email.lower()
            record.display_name, record.organization_id = user.display_name, pack.organization_id
            record.department_ids_json = json.dumps(user.departments)
            record.workspace_ids_json = json.dumps(user.workspaces)
            record.roles_json = json.dumps([item.value for item in user.roles])
            record.permissions_json = json.dumps([item.value for item in user.permissions])
            record.clearance = user.clearance.name.upper()
            record.enabled = user.enabled if local else False
            self.session.add(record)
        self.session.flush()

    @staticmethod
    def _rule_map(rules: list[AccessRule]) -> dict[str, AccessRule]:
        return {item.id: item for item in rules}

    def _upsert_assets(self, pack: LoadedOrganizationPack) -> None:
        rules = self._rule_map(pack.access_matrix.asset_rules)
        for asset in pack.assets_file.assets:
            access = rules[asset.access_rule_id]
            record = self.session.get(AssetRecord, asset.id) or AssetRecord(
                id=asset.id, canonical_name=asset.canonical_name, asset_type=asset.asset_type,
                organization_id=pack.organization_id, plant_id=asset.plant_id,
                area_id=asset.area_id, unit_id=asset.unit_id,
                workspace_id=access.workspace_id, criticality=asset.criticality,
                status=asset.status,
            )
            record.canonical_name, record.asset_type = asset.canonical_name, asset.asset_type
            record.organization_id, record.plant_id = pack.organization_id, asset.plant_id
            record.area_id, record.unit_id, record.workspace_id = asset.area_id, asset.unit_id, access.workspace_id
            record.department_id = access.department_id
            record.classification = access.classification.name.upper()
            record.allowed_roles_json = json.dumps([item.value for item in access.allowed_roles])
            record.allowed_users_json = json.dumps(access.allowed_users)
            record.owner_id, record.criticality, record.status = access.owner_id, asset.criticality, asset.status
            record.manufacturer, record.model, record.commissioned_at = asset.manufacturer, asset.model, asset.commissioned_at
            record.design_parameters_json, record.tags_json = json.dumps(asset.design_parameters), json.dumps(asset.tags)
            self.session.add(record)
            self.session.query(AssetAliasRecord).filter_by(asset_id=asset.id).delete(synchronize_session=False)
            for alias in [asset.id, *asset.aliases]:
                self.session.add(AssetAliasRecord(
                    id=str(uuid4()), asset_id=asset.id, alias=alias,
                    alias_normalized=" ".join(alias.strip().upper().split()),
                ))
        self.session.flush()

    def _ingest_documents(self, pack: LoadedOrganizationPack) -> None:
        rules = self._rule_map(pack.access_matrix.document_rules)
        root = Path(pack.root) / "documents"
        ingestion = KnowledgeIngestionService(
            self.session, self.embeddings or configured_embedding_provider(),
        )
        for item in pack.documents_file.documents:
            access = rules[item.access_rule_id]
            acl = DocumentACL(
                organization_id=pack.organization_id, department_id=access.department_id,
                workspace_id=access.workspace_id, classification=access.classification,
                allowed_roles=access.allowed_roles, allowed_users=access.allowed_users,
                owner_id=access.owner_id,
            )
            metadata = {
                **item.metadata,
                "revision": item.revision,
                "approval_status": item.approval_status,
                "asset_id": item.asset_ids[0] if len(item.asset_ids) == 1 else None,
                "asset_ids": item.asset_ids,
                "organization_pack_fingerprint": pack.pack_fingerprint,
            }
            document = ingestion.ingest(
                root / item.path, metadata, acl=acl, require_acl=True, commit=False,
            )
            for asset_id in item.asset_ids:
                existing = self.session.query(AssetEvidenceLinkRecord).filter_by(
                    asset_id=asset_id, evidence_id=document.id, relationship=item.relationship,
                ).first()
                if not existing:
                    self.session.add(AssetEvidenceLinkRecord(
                        id=str(uuid4()), asset_id=asset_id, evidence_id=document.id,
                        relationship=item.relationship, source="organization-pack",
                        confidence=1.0, inferred=False,
                    ))
        self.session.flush()

    def _install_workcells(
        self, pack: LoadedOrganizationPack, statuses: list[WorkcellPackStatus],
    ) -> list[Path]:
        installed: list[Path] = []
        root = self.settings.workcells_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        try:
            for source_raw, status in zip(pack.workcell_directories, statuses):
                if not status.valid or not status.workcell_id or not status.version:
                    continue
                safe_version = re.sub(r"[^A-Za-z0-9._-]", "-", status.version)
                destination = (root / f"{pack.organization_id}--{status.workcell_id}--{safe_version}").resolve()
                if root not in destination.parents:
                    raise OrganizationImportError("Workcell destination escaped configured root")
                if destination.exists():
                    existing = WorkcellLoader(root).load(destination)
                    if existing.content_hash != status.content_hash:
                        raise OrganizationImportError(f"Installed Workcell differs: {status.workcell_id}")
                    continue
                shutil.copytree(Path(source_raw), destination)
                installed.append(destination)
        except Exception:
            for path in installed:
                if root in path.resolve().parents and path.is_dir():
                    shutil.rmtree(path)
            raise
        return installed

    def _write_report(self, report: OrganizationImportReport) -> OrganizationImportReport:
        root = self.settings.organization_reports_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = root / f"{report.organization_id}-{stamp}-{report.pack_fingerprint[:12]}.json"
        report.report_path = str(path)
        path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return report
