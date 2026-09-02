from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CapsuleStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CapsuleState(str, Enum):
    building = "BUILDING"
    complete = "COMPLETE"
    verified = "VERIFIED"
    invalid = "INVALID"
    failed = "FAILED"


class CapsuleStatus(str, Enum):
    valid = "VALID"
    invalid = "INVALID"


class SignatureStatus(str, Enum):
    valid = "VALID"
    unsigned = "UNSIGNED"
    signed_unverified = "SIGNED_UNVERIFIED"
    invalid = "INVALID"


class CapsuleFileIdentity(CapsuleStrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)
    category: str


class CapsuleWorkcellIdentity(CapsuleStrictModel):
    id: str
    version: str
    hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    workflow_version: str


class CapsuleManifest(CapsuleStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    capsule_id: str
    task_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_summary: str
    workcell: CapsuleWorkcellIdentity
    models: list[dict[str, Any]] = Field(default_factory=list)
    inputs: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    files: list[CapsuleFileIdentity]
    capsule_root_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    root_hash_algorithm: Literal["sha256-canonical-path-hash-list-v1"] = "sha256-canonical-path-hash-list-v1"


class CapsuleSignature(CapsuleStrictModel):
    algorithm: str
    key_id: str
    signed_root_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    signature: str


class CapsuleVerificationFailure(CapsuleStrictModel):
    path: str | None = None
    type: str
    message: str


class CapsuleVerificationResult(CapsuleStrictModel):
    status: CapsuleStatus
    manifest_valid: bool
    hashes_valid: bool
    root_hash_valid: bool
    signature_status: SignatureStatus
    signature_valid: bool | None = None
    artifact_count: int = 0
    artifact_valid_count: int = 0
    failures: list[CapsuleVerificationFailure] = Field(default_factory=list)
