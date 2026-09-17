from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from app.organizations.models import (
    AccessMatrixFile, AssetsFile, DocumentsFile, LoadedOrganizationPack,
    OrganizationFile, OrganizationPackValidation, PackValidationIssue, UsersFile,
)


class OrganizationPackError(ValueError):
    pass


class OrganizationPackLoader:
    """Loads an immutable-on-read organization pack from a bounded local root."""

    ROOT_FILES = {"organization.yaml", "assets.yaml", "access_matrix.yaml", "users.yaml"}
    ROOT_DIRECTORIES = {"documents", "workcells"}
    DOCUMENT_SUFFIXES = {".txt", ".md", ".csv", ".pdf", ".docx", ".xlsx", ".json"}

    def __init__(
        self, root: Path, *, max_file_bytes: int = 25 * 1024 * 1024,
        max_total_bytes: int = 500 * 1024 * 1024, max_files: int = 10_000,
    ) -> None:
        self.root = root.resolve()
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        self.max_files = max_files

    @staticmethod
    def _read_yaml(path: Path) -> Any:
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise OrganizationPackError(f"Cannot read {path.name}: {exc}") from exc

    @staticmethod
    def _duplicate_values(values: list[str]) -> list[str]:
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return sorted(value for value, count in counts.items() if count > 1)

    def _contained(self, path: Path) -> Path:
        if path.is_symlink():
            raise OrganizationPackError(f"Symlinks are not allowed: {path}")
        resolved = path.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise OrganizationPackError(f"Path escapes Organization Pack: {path}")
        return resolved

    def _inventory(self) -> list[Path]:
        if not self.root.is_dir():
            raise OrganizationPackError(f"Organization Pack directory does not exist: {self.root}")
        files: list[Path] = []
        total = 0
        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root)
            if any(part.startswith(".") for part in relative.parts):
                raise OrganizationPackError(f"Hidden pack content is not allowed: {relative.as_posix()}")
            self._contained(path)
            if path.is_dir():
                if relative.parts[0] not in self.ROOT_DIRECTORIES:
                    raise OrganizationPackError(f"Unknown pack directory: {relative.parts[0]}")
                continue
            if len(relative.parts) == 1 and relative.name not in self.ROOT_FILES:
                raise OrganizationPackError(f"Unknown pack root file: {relative.name}")
            size = path.stat().st_size
            if size > self.max_file_bytes:
                raise OrganizationPackError(f"Pack file exceeds size limit: {relative.as_posix()}")
            total += size
            files.append(path)
            if len(files) > self.max_files or total > self.max_total_bytes:
                raise OrganizationPackError("Organization Pack exceeds bounded file or byte limits")
        return files

    def _fingerprint(self, files: list[Path]) -> str:
        digest = hashlib.sha256()
        for path in sorted(files, key=lambda item: item.relative_to(self.root).as_posix()):
            digest.update(path.relative_to(self.root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()

    def load(self) -> LoadedOrganizationPack:
        files = self._inventory()
        required = ["organization.yaml", "assets.yaml", "access_matrix.yaml", "documents/manifest.yaml"]
        missing = [name for name in required if not (self.root / name).is_file()]
        if missing:
            raise OrganizationPackError(f"Required pack files are missing: {missing}")
        try:
            organization = OrganizationFile.model_validate(self._read_yaml(self.root / "organization.yaml"))
            access = AccessMatrixFile.model_validate(self._read_yaml(self.root / "access_matrix.yaml"))
            assets = AssetsFile.model_validate(self._read_yaml(self.root / "assets.yaml"))
            users_path = self.root / "users.yaml"
            users = UsersFile.model_validate(
                self._read_yaml(users_path) if users_path.exists()
                else {"schema_version": "1.0", "identity_provider": "external", "users": []}
            )
            documents = DocumentsFile.model_validate(self._read_yaml(self.root / "documents/manifest.yaml"))
        except ValidationError as exc:
            raise OrganizationPackError(str(exc)) from exc
        workcell_root = self.root / "workcells"
        workcells = sorted(
            str(path.resolve()) for path in workcell_root.iterdir() if path.is_dir()
        ) if workcell_root.is_dir() else []
        loaded = LoadedOrganizationPack(
            root=str(self.root), organization_file=organization, access_matrix=access,
            assets_file=assets, users_file=users, documents_file=documents,
            pack_fingerprint=self._fingerprint(files), workcell_directories=workcells,
        )
        validation = self.validate(loaded)
        if not validation.valid:
            raise OrganizationPackError(json.dumps(validation.model_dump(mode="json"), indent=2))
        return loaded

    def validate(self, pack: LoadedOrganizationPack) -> OrganizationPackValidation:
        issues: list[PackValidationIssue] = []

        def issue(code: str, message: str, path: str) -> None:
            issues.append(PackValidationIssue(code=code, message=message, path=path))

        organization = pack.organization_file
        ids_by_group = {
            "plants": [item.id for item in organization.plants],
            "areas": [item.id for item in organization.areas],
            "units": [item.id for item in organization.units],
            "departments": [item.id for item in organization.departments],
            "workspaces": [item.id for item in organization.workspaces],
            "assets": [item.id for item in pack.assets_file.assets],
            "users": [item.id for item in pack.users_file.users],
        }
        for group, values in ids_by_group.items():
            duplicates = self._duplicate_values(values)
            if duplicates:
                issue("DUPLICATE_ID", f"Duplicate {group}: {duplicates}", f"{group}.yaml")
        plants, areas, units = map(set, (ids_by_group["plants"], ids_by_group["areas"], ids_by_group["units"]))
        departments, workspaces = set(ids_by_group["departments"]), set(ids_by_group["workspaces"])
        users, assets = set(ids_by_group["users"]), set(ids_by_group["assets"])
        area_map = {item.id: item for item in organization.areas}
        for area in organization.areas:
            if area.plant_id not in plants:
                issue("UNKNOWN_PLANT", f"Area {area.id} references {area.plant_id}", "organization.yaml")
        for unit in organization.units:
            if unit.plant_id not in plants or unit.area_id not in areas:
                issue("UNKNOWN_LOCATION", f"Unit {unit.id} has an unknown plant or area", "organization.yaml")
            elif area_map[unit.area_id].plant_id != unit.plant_id:
                issue("LOCATION_MISMATCH", f"Unit {unit.id} plant does not match its area", "organization.yaml")
        all_rules = [*pack.access_matrix.asset_rules, *pack.access_matrix.document_rules]
        rule_ids = [item.id for item in all_rules]
        duplicates = self._duplicate_values(rule_ids)
        if duplicates:
            issue("DUPLICATE_ACCESS_RULE", f"Duplicate access rules: {duplicates}", "access_matrix.yaml")
        for rule in all_rules:
            if rule.workspace_id not in workspaces or (rule.department_id and rule.department_id not in departments):
                issue("UNKNOWN_ACCESS_SCOPE", f"Access rule {rule.id} references an unknown scope", "access_matrix.yaml")
            unknown_users = set(rule.allowed_users) - users
            if unknown_users:
                issue("UNKNOWN_ALLOWED_USER", f"Access rule {rule.id} references {sorted(unknown_users)}", "access_matrix.yaml")
            if rule.owner_id and rule.owner_id not in users:
                issue("UNKNOWN_OWNER", f"Access rule {rule.id} references owner {rule.owner_id}", "access_matrix.yaml")
        asset_rules = {item.id: item for item in pack.access_matrix.asset_rules}
        document_rules = {item.id: item for item in pack.access_matrix.document_rules}
        alias_owners: dict[str, str] = {}
        for asset in pack.assets_file.assets:
            if asset.plant_id not in plants or asset.area_id not in areas or asset.unit_id not in units:
                issue("UNKNOWN_ASSET_LOCATION", f"Asset {asset.id} references an unknown location", "assets.yaml")
            if asset.access_rule_id not in asset_rules:
                issue("UNKNOWN_ASSET_ACCESS_RULE", f"Asset {asset.id} references {asset.access_rule_id}", "assets.yaml")
            for alias in [asset.id, *asset.aliases]:
                normalized = " ".join(alias.strip().upper().split())
                owner = alias_owners.setdefault(normalized, asset.id)
                if owner != asset.id:
                    issue("AMBIGUOUS_ASSET_ALIAS", f"Alias {alias!r} belongs to both {owner} and {asset.id}", "assets.yaml")
        for user in pack.users_file.users:
            if set(user.departments) - departments or set(user.workspaces) - workspaces:
                issue("UNKNOWN_USER_SCOPE", f"User {user.id} references an unknown department or workspace", "users.yaml")
        seen_paths: set[str] = set()
        root = Path(pack.root)
        for document in pack.documents_file.documents:
            if document.path in seen_paths:
                issue("DUPLICATE_DOCUMENT", f"Document listed more than once: {document.path}", "documents/manifest.yaml")
            seen_paths.add(document.path)
            path = root / "documents" / document.path
            try:
                resolved = self._contained(path)
            except OrganizationPackError as exc:
                issue("UNSAFE_DOCUMENT_PATH", str(exc), document.path)
                continue
            if not resolved.is_file():
                issue("DOCUMENT_MISSING", f"Document does not exist: {document.path}", document.path)
            elif resolved.suffix.lower() not in self.DOCUMENT_SUFFIXES:
                issue("DOCUMENT_TYPE_UNSUPPORTED", f"Unsupported document type: {resolved.suffix}", document.path)
            if document.access_rule_id not in document_rules:
                issue("UNKNOWN_DOCUMENT_ACCESS_RULE", f"Document {document.path} references {document.access_rule_id}", "documents/manifest.yaml")
            unknown_assets = set(document.asset_ids) - assets
            if unknown_assets:
                issue("UNKNOWN_DOCUMENT_ASSET", f"Document {document.path} references {sorted(unknown_assets)}", "documents/manifest.yaml")
        secrets = sorted({item.password_env for item in pack.users_file.users if item.password_env})
        counts = {
            **{key: len(value) for key, value in ids_by_group.items()},
            "documents": len(pack.documents_file.documents),
            "workcells": len(pack.workcell_directories),
        }
        return OrganizationPackValidation(
            valid=not issues, organization_id=pack.organization_id,
            pack_fingerprint=pack.pack_fingerprint, issues=issues,
            required_secret_environment_variables=secrets, counts=counts,
        )
