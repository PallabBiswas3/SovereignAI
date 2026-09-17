from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.identity.models import ClearanceLevel, Permission, Role


PACK_SCHEMA_VERSION = "1.0"
Identifier = str


class StrictPackModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NamedEntity(StrictPackModel):
    id: Identifier = Field(min_length=2, max_length=100, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    name: str = Field(min_length=2, max_length=255)


class OrganizationDefinition(NamedEntity):
    short_name: str = Field(min_length=2, max_length=40)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlantDefinition(NamedEntity):
    pass


class AreaDefinition(NamedEntity):
    plant_id: Identifier


class UnitDefinition(NamedEntity):
    plant_id: Identifier
    area_id: Identifier


class OrganizationFile(StrictPackModel):
    schema_version: Literal["1.0"]
    organization: OrganizationDefinition
    plants: list[PlantDefinition] = Field(min_length=1)
    areas: list[AreaDefinition] = Field(default_factory=list)
    units: list[UnitDefinition] = Field(default_factory=list)
    departments: list[NamedEntity] = Field(min_length=1)
    workspaces: list[NamedEntity] = Field(min_length=1)


class AccessRule(StrictPackModel):
    id: Identifier = Field(min_length=2, max_length=100, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    workspace_id: Identifier
    department_id: Identifier | None = None
    classification: ClearanceLevel = ClearanceLevel.internal
    allowed_roles: list[Role] = Field(default_factory=list)
    allowed_users: list[Identifier] = Field(default_factory=list)
    owner_id: Identifier | None = None

    @field_validator("classification", mode="before")
    @classmethod
    def parse_classification(cls, value):
        return ClearanceLevel.parse(value)


class AccessMatrixFile(StrictPackModel):
    schema_version: Literal["1.0"]
    asset_rules: list[AccessRule] = Field(min_length=1)
    document_rules: list[AccessRule] = Field(min_length=1)


class AssetDefinition(StrictPackModel):
    id: Identifier = Field(min_length=2, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    canonical_name: str = Field(min_length=2, max_length=200)
    asset_type: str = Field(min_length=2, max_length=100)
    plant_id: Identifier
    area_id: Identifier
    unit_id: Identifier
    access_rule_id: Identifier
    criticality: str = Field(default="MEDIUM", min_length=2, max_length=30)
    status: str = Field(default="IN_SERVICE", min_length=2, max_length=40)
    manufacturer: str | None = Field(default=None, max_length=200)
    model: str | None = Field(default=None, max_length=160)
    commissioned_at: datetime | None = None
    design_parameters: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)

    @field_validator("aliases")
    @classmethod
    def unique_aliases(cls, value: list[str]) -> list[str]:
        normalized = [item.strip().upper() for item in value]
        if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("Asset aliases must be non-empty and unique")
        return value


class AssetsFile(StrictPackModel):
    schema_version: Literal["1.0"]
    assets: list[AssetDefinition] = Field(default_factory=list)


class UserDefinition(StrictPackModel):
    id: Identifier = Field(min_length=2, max_length=100, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    email: str = Field(min_length=3, max_length=255)
    display_name: str = Field(min_length=2, max_length=160)
    departments: list[Identifier] = Field(default_factory=list)
    workspaces: list[Identifier] = Field(min_length=1)
    roles: list[Role] = Field(min_length=1)
    clearance: ClearanceLevel = ClearanceLevel.internal
    permissions: list[Permission] = Field(default_factory=list)
    enabled: bool = True
    password_env: str | None = Field(
        default=None, pattern=r"^[A-Z][A-Z0-9_]{2,127}$",
        description="Environment variable containing a local bootstrap password; never the password itself.",
    )

    @field_validator("clearance", mode="before")
    @classmethod
    def parse_clearance(cls, value):
        return ClearanceLevel.parse(value)

    @field_validator("email")
    @classmethod
    def basic_email_validation(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized.count("@") != 1 or "." not in normalized.split("@", 1)[1]:
            raise ValueError("A valid email address is required")
        return normalized


class UsersFile(StrictPackModel):
    schema_version: Literal["1.0"]
    identity_provider: Literal["local", "external"] = "external"
    users: list[UserDefinition] = Field(default_factory=list)

    @model_validator(mode="after")
    def local_password_references_required(self):
        if self.identity_provider == "local":
            missing = [item.id for item in self.users if item.enabled and not item.password_env]
            if missing:
                raise ValueError(f"Enabled local users require password_env: {missing}")
        return self


class DocumentDefinition(StrictPackModel):
    path: str = Field(min_length=1, max_length=500)
    access_rule_id: Identifier
    revision: str | None = Field(default=None, max_length=80)
    approval_status: Literal["APPROVED", "DRAFT", "SUPERSEDED", "UNKNOWN"]
    asset_ids: list[Identifier] = Field(default_factory=list)
    relationship: str = Field(default="HAS_DOCUMENT", min_length=2, max_length=60)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        normalized = value.replace("\\", "/").strip()
        parts = normalized.split("/")
        if normalized.startswith("/") or ":" in parts[0] or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("Document path must be a safe relative path")
        return normalized


class DocumentsFile(StrictPackModel):
    schema_version: Literal["1.0"]
    documents: list[DocumentDefinition] = Field(default_factory=list)


class PackValidationIssue(StrictPackModel):
    code: str
    message: str
    path: str | None = None


class WorkcellPackStatus(StrictPackModel):
    directory: str
    workcell_id: str | None = None
    version: str | None = None
    content_hash: str | None = None
    valid: bool
    status: str
    issues: list[str] = Field(default_factory=list)


class OrganizationPackValidation(StrictPackModel):
    valid: bool
    organization_id: str | None = None
    pack_fingerprint: str | None = None
    issues: list[PackValidationIssue] = Field(default_factory=list)
    required_secret_environment_variables: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    workcells: list[WorkcellPackStatus] = Field(default_factory=list)


class LoadedOrganizationPack(StrictPackModel):
    root: str
    organization_file: OrganizationFile
    access_matrix: AccessMatrixFile
    assets_file: AssetsFile
    users_file: UsersFile
    documents_file: DocumentsFile
    pack_fingerprint: str
    workcell_directories: list[str] = Field(default_factory=list)

    @property
    def organization_id(self) -> str:
        return self.organization_file.organization.id


class OrganizationImportReport(StrictPackModel):
    status: Literal["DRY_RUN", "IMPORTED"]
    organization_id: str
    pack_fingerprint: str
    counts: dict[str, int]
    workcells: list[WorkcellPackStatus] = Field(default_factory=list)
    required_secret_environment_variables: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    report_path: str | None = None
