from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class OrganizationRecord(Base):
    __tablename__ = "organizations"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    short_name: Mapped[str] = mapped_column(String(40))
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


class DepartmentRecord(Base):
    __tablename__ = "departments"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(String(160))


class WorkspaceRecord(Base):
    __tablename__ = "workspaces"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(String(160))


class UserRecord(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    email_normalized: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(160))
    organization_id: Mapped[str] = mapped_column(String(100), index=True)
    department_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    workspace_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    roles_json: Mapped[str] = mapped_column(Text, default="[]")
    permissions_json: Mapped[str] = mapped_column(Text, default="[]")
    clearance: Mapped[str] = mapped_column(String(30), default="INTERNAL")
    password_hash: Mapped[str] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class IdentitySessionRecord(Base):
    __tablename__ = "identity_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(100), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    csrf_token: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TaskAccessRecord(Base):
    __tablename__ = "task_access"
    task_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    organization_id: Mapped[str] = mapped_column(String(100), index=True)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    workspace_id: Mapped[str] = mapped_column(String(100), index=True)
    department_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    classification: Mapped[str] = mapped_column(String(30), default="INTERNAL")


class PlantRecord(Base):
    __tablename__ = "plants"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(String(200))


class AreaRecord(Base):
    __tablename__ = "plant_areas"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(100), index=True)
    plant_id: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(String(200))


class UnitRecord(Base):
    __tablename__ = "plant_units"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(100), index=True)
    plant_id: Mapped[str] = mapped_column(String(100), index=True)
    area_id: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(String(200))


class AssetRecord(Base):
    __tablename__ = "assets"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(200))
    asset_type: Mapped[str] = mapped_column(String(100), index=True)
    organization_id: Mapped[str] = mapped_column(String(100), index=True)
    plant_id: Mapped[str] = mapped_column(String(100), index=True)
    area_id: Mapped[str] = mapped_column(String(100), index=True)
    unit_id: Mapped[str] = mapped_column(String(100), index=True)
    workspace_id: Mapped[str] = mapped_column(String(100), index=True)
    department_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    classification: Mapped[str] = mapped_column(String(30), default="INTERNAL")
    allowed_roles_json: Mapped[str] = mapped_column(Text, default="[]")
    allowed_users_json: Mapped[str] = mapped_column(Text, default="[]")
    owner_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    criticality: Mapped[str] = mapped_column(String(30))
    status: Mapped[str] = mapped_column(String(40), index=True)
    manufacturer: Mapped[str | None] = mapped_column(String(200), nullable=True)
    model: Mapped[str | None] = mapped_column(String(160), nullable=True)
    commissioned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    design_parameters_json: Mapped[str] = mapped_column(Text, default="{}")
    tags_json: Mapped[str] = mapped_column(Text, default="[]")


class AssetAliasRecord(Base):
    __tablename__ = "asset_aliases"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(100), index=True)
    alias: Mapped[str] = mapped_column(String(160), index=True)
    alias_normalized: Mapped[str] = mapped_column(String(160), index=True)


class AssetEvidenceLinkRecord(Base):
    __tablename__ = "asset_evidence_links"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(100), index=True)
    evidence_id: Mapped[str] = mapped_column(String(100), index=True)
    relationship: Mapped[str] = mapped_column(String(60), index=True)
    source: Mapped[str] = mapped_column(String(160))
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    inferred: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class InspectionRecordRow(Base):
    __tablename__ = "asset_inspections"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(100), index=True)
    inspected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source_document_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    summary: Mapped[str] = mapped_column(Text)
    measurement_ids_json: Mapped[str] = mapped_column(Text, default="[]")


class OperationalMeasurementRecord(Base):
    __tablename__ = "operational_measurements"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(100), index=True)
    metric: Mapped[str] = mapped_column(String(100), index=True)
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(40))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    quality: Mapped[str] = mapped_column(String(30), index=True)
    source: Mapped[str] = mapped_column(String(160), index=True)
    source_tag: Mapped[str] = mapped_column(String(300))
    original_value: Mapped[float] = mapped_column(Float)
    original_unit: Mapped[str] = mapped_column(String(40))
    scenario: Mapped[str] = mapped_column(String(80), default="PUMP_102_DEGRADING", index=True)


class MaintenanceRecordRow(Base):
    __tablename__ = "maintenance_records"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(100), index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    event_type: Mapped[str] = mapped_column(String(80))
    title: Mapped[str] = mapped_column(String(240))
    summary: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(160))
    status: Mapped[str] = mapped_column(String(40), default="COMPLETED")


class MaintenanceDraftRecord(Base):
    __tablename__ = "maintenance_drafts"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(100), index=True)
    action_type: Mapped[str] = mapped_column(String(80))
    priority: Mapped[str] = mapped_column(String(30))
    title: Mapped[str] = mapped_column(String(240))
    description: Mapped[str] = mapped_column(Text)
    reason_claim_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    status: Mapped[str] = mapped_column(String(30), default="DRAFT", index=True)
    created_by: Mapped[str] = mapped_column(String(100), index=True)
    organization_id: Mapped[str] = mapped_column(String(100), index=True)
    workspace_id: Mapped[str] = mapped_column(String(100), index=True)
    department_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    classification: Mapped[str] = mapped_column(String(30), default="CONFIDENTIAL")
    approval_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(36), index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class AgentRunRecord(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30), index=True)
    state_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    organization_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    owner_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    workspace_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    department_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    classification: Mapped[str] = mapped_column(String(30), default="INTERNAL")


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), index=True)
    checksum: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_provider: Mapped[str] = mapped_column(String(255), default="local-feature-hash")
    embedding_dimension: Mapped[int] = mapped_column(Integer, default=384)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    organization_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    owner_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    workspace_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    department_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    classification: Mapped[str] = mapped_column(String(30), default="INTERNAL")
    allowed_roles_json: Mapped[str] = mapped_column(Text, default="[]")
    allowed_users_json: Mapped[str] = mapped_column(Text, default="[]")


class KnowledgeChunkRecord(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("knowledge_documents.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(String(200), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    embedding_json: Mapped[str] = mapped_column(Text)


class ArtifactRecord(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    media_type: Mapped[str] = mapped_column(String(120))
    path: Mapped[str] = mapped_column(Text)
    size: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    workcell_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    workcell_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    artifact_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    lineage_json: Mapped[str] = mapped_column(Text, default="{}")
    organization_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    owner_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    workspace_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    department_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    classification: Mapped[str] = mapped_column(String(30), default="INTERNAL")
    allowed_roles_json: Mapped[str] = mapped_column(Text, default="[]")
    allowed_users_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class AuditEventRecord(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    summary: Mapped[str] = mapped_column(String(500))
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    principal_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    organization_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class HumanApprovalRecord(Base):
    __tablename__ = "human_approvals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    tool: Mapped[str] = mapped_column(String(100))
    args_json: Mapped[str] = mapped_column(Text)
    risk: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(30), default="pending")
    decided_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    execution_status: Mapped[str | None] = mapped_column(String(30), nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    requester_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    organization_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    workspace_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    action_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class NetworkEventRecord(Base):
    __tablename__ = "network_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    destination: Mapped[str] = mapped_column(String(500))
    component: Mapped[str] = mapped_column(String(120))
    allowed: Mapped[bool] = mapped_column(Boolean, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class MetricSnapshotRecord(Base):
    __tablename__ = "metric_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    metrics_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class TaskEventRecord(Base):
    __tablename__ = "task_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(36), index=True)
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class CacheRecord(Base):
    __tablename__ = "cache_entries"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(40), index=True)
    cache_key: Mapped[str] = mapped_column(String(64), index=True)
    value_json: Mapped[str] = mapped_column(Text, default="null")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_accessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EvidenceCapsuleRecord(Base):
    __tablename__ = "evidence_capsules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(36), index=True)
    state: Mapped[str] = mapped_column(String(30), index=True)
    path: Mapped[str] = mapped_column(Text)
    capsule_root_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    signature_status: Mapped[str] = mapped_column(String(40), default="UNSIGNED")
    manifest_json: Mapped[str] = mapped_column(Text, default="{}")
    organization_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    owner_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    workspace_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    department_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    classification: Mapped[str] = mapped_column(String(30), default="INTERNAL")
    allowed_roles_json: Mapped[str] = mapped_column(Text, default="[]")
    allowed_users_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))





settings = get_settings()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    if engine.dialect.name == "sqlite":
        columns = {column["name"] for column in inspect(engine).get_columns("knowledge_documents")}
        migrations = []
        if "embedding_provider" not in columns:
            migrations.append("ALTER TABLE knowledge_documents ADD COLUMN embedding_provider VARCHAR(255) NOT NULL DEFAULT 'local-feature-hash'")
        if "embedding_dimension" not in columns:
            migrations.append("ALTER TABLE knowledge_documents ADD COLUMN embedding_dimension INTEGER NOT NULL DEFAULT 384")
        if migrations:
            with engine.begin() as connection:
                for statement in migrations:
                    connection.execute(text(statement))
        approval_columns = {column["name"] for column in inspect(engine).get_columns("human_approvals")}
        approval_migrations = []
        if "execution_status" not in approval_columns:
            approval_migrations.append("ALTER TABLE human_approvals ADD COLUMN execution_status VARCHAR(30)")
        if "result_json" not in approval_columns:
            approval_migrations.append("ALTER TABLE human_approvals ADD COLUMN result_json TEXT")
        if "executed_at" not in approval_columns:
            approval_migrations.append("ALTER TABLE human_approvals ADD COLUMN executed_at DATETIME")
        if approval_migrations:
            with engine.begin() as connection:
                for statement in approval_migrations:
                    connection.execute(text(statement))
        artifact_columns = {column["name"] for column in inspect(engine).get_columns("artifacts")}
        artifact_migrations = []
        if "sha256" not in artifact_columns:
            artifact_migrations.append("ALTER TABLE artifacts ADD COLUMN sha256 VARCHAR(64)")
        if "workcell_id" not in artifact_columns:
            artifact_migrations.append("ALTER TABLE artifacts ADD COLUMN workcell_id VARCHAR(100)")
        if "workcell_version" not in artifact_columns:
            artifact_migrations.append("ALTER TABLE artifacts ADD COLUMN workcell_version VARCHAR(40)")
        if "artifact_type" not in artifact_columns:
            artifact_migrations.append("ALTER TABLE artifacts ADD COLUMN artifact_type VARCHAR(80)")
        if "lineage_json" not in artifact_columns:
            artifact_migrations.append("ALTER TABLE artifacts ADD COLUMN lineage_json TEXT NOT NULL DEFAULT '{}'")
        if artifact_migrations:
            with engine.begin() as connection:
                for statement in artifact_migrations:
                    connection.execute(text(statement))
        scoped_migrations = {
            "agent_runs": {
                "organization_id": "ALTER TABLE agent_runs ADD COLUMN organization_id VARCHAR(100)",
                "owner_id": "ALTER TABLE agent_runs ADD COLUMN owner_id VARCHAR(100)",
                "workspace_id": "ALTER TABLE agent_runs ADD COLUMN workspace_id VARCHAR(100)",
                "department_id": "ALTER TABLE agent_runs ADD COLUMN department_id VARCHAR(100)",
                "classification": "ALTER TABLE agent_runs ADD COLUMN classification VARCHAR(30) NOT NULL DEFAULT 'INTERNAL'",
            },
            "knowledge_documents": {
                "organization_id": "ALTER TABLE knowledge_documents ADD COLUMN organization_id VARCHAR(100)",
                "owner_id": "ALTER TABLE knowledge_documents ADD COLUMN owner_id VARCHAR(100)",
                "workspace_id": "ALTER TABLE knowledge_documents ADD COLUMN workspace_id VARCHAR(100)",
                "department_id": "ALTER TABLE knowledge_documents ADD COLUMN department_id VARCHAR(100)",
                "classification": "ALTER TABLE knowledge_documents ADD COLUMN classification VARCHAR(30) NOT NULL DEFAULT 'INTERNAL'",
                "allowed_roles_json": "ALTER TABLE knowledge_documents ADD COLUMN allowed_roles_json TEXT NOT NULL DEFAULT '[]'",
                "allowed_users_json": "ALTER TABLE knowledge_documents ADD COLUMN allowed_users_json TEXT NOT NULL DEFAULT '[]'",
            },
            "artifacts": {
                "organization_id": "ALTER TABLE artifacts ADD COLUMN organization_id VARCHAR(100)",
                "owner_id": "ALTER TABLE artifacts ADD COLUMN owner_id VARCHAR(100)",
                "workspace_id": "ALTER TABLE artifacts ADD COLUMN workspace_id VARCHAR(100)",
                "department_id": "ALTER TABLE artifacts ADD COLUMN department_id VARCHAR(100)",
                "classification": "ALTER TABLE artifacts ADD COLUMN classification VARCHAR(30) NOT NULL DEFAULT 'INTERNAL'",
                "allowed_roles_json": "ALTER TABLE artifacts ADD COLUMN allowed_roles_json TEXT NOT NULL DEFAULT '[]'",
                "allowed_users_json": "ALTER TABLE artifacts ADD COLUMN allowed_users_json TEXT NOT NULL DEFAULT '[]'",
            },
            "evidence_capsules": {
                "organization_id": "ALTER TABLE evidence_capsules ADD COLUMN organization_id VARCHAR(100)",
                "owner_id": "ALTER TABLE evidence_capsules ADD COLUMN owner_id VARCHAR(100)",
                "workspace_id": "ALTER TABLE evidence_capsules ADD COLUMN workspace_id VARCHAR(100)",
                "department_id": "ALTER TABLE evidence_capsules ADD COLUMN department_id VARCHAR(100)",
                "classification": "ALTER TABLE evidence_capsules ADD COLUMN classification VARCHAR(30) NOT NULL DEFAULT 'INTERNAL'",
                "allowed_roles_json": "ALTER TABLE evidence_capsules ADD COLUMN allowed_roles_json TEXT NOT NULL DEFAULT '[]'",
                "allowed_users_json": "ALTER TABLE evidence_capsules ADD COLUMN allowed_users_json TEXT NOT NULL DEFAULT '[]'",
            },
            "human_approvals": {
                "requester_id": "ALTER TABLE human_approvals ADD COLUMN requester_id VARCHAR(100)",
                "organization_id": "ALTER TABLE human_approvals ADD COLUMN organization_id VARCHAR(100)",
                "workspace_id": "ALTER TABLE human_approvals ADD COLUMN workspace_id VARCHAR(100)",
                "action_hash": "ALTER TABLE human_approvals ADD COLUMN action_hash VARCHAR(64)",
            },
            "audit_events": {
                "principal_id": "ALTER TABLE audit_events ADD COLUMN principal_id VARCHAR(100)",
                "organization_id": "ALTER TABLE audit_events ADD COLUMN organization_id VARCHAR(100)",
            },
        }
        pending: list[str] = []
        for table, statements in scoped_migrations.items():
            existing = {column["name"] for column in inspect(engine).get_columns(table)}
            pending.extend(statement for column, statement in statements.items() if column not in existing)
        if pending:
            with engine.begin() as connection:
                for statement in pending:
                    connection.execute(text(statement))


def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
