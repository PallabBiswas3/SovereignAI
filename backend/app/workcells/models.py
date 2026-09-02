from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


WORKCELL_PLATFORM_VERSION = "2.0.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkcellStatus(str, Enum):
    ready = "READY"
    invalid = "INVALID"
    incompatible = "INCOMPATIBLE"
    disabled = "DISABLED"
    missing_dependency = "MISSING_DEPENDENCY"
    untrusted = "UNTRUSTED"


class WorkcellTrustStatus(str, Enum):
    unsigned = "UNSIGNED"
    signed_unverified = "SIGNED_UNVERIFIED"
    trusted = "TRUSTED"
    invalid_signature = "INVALID_SIGNATURE"
    modified = "MODIFIED"


class WorkcellFailureCode(str, Enum):
    not_found = "WORKCELL_NOT_FOUND"
    invalid = "WORKCELL_INVALID"
    incompatible = "WORKCELL_INCOMPATIBLE"
    disabled = "WORKCELL_DISABLED"
    untrusted = "WORKCELL_UNTRUSTED"
    tool_not_allowed = "WORKCELL_TOOL_NOT_ALLOWED"
    handler_not_found = "WORKCELL_HANDLER_NOT_FOUND"
    input_invalid = "WORKCELL_INPUT_INVALID"
    execution_failed = "WORKCELL_EXECUTION_FAILED"


class WorkcellToolRequirement(StrictModel):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_]*$")
    required: bool = True


class WorkcellEvidenceRequirement(StrictModel):
    id: str = Field(min_length=1, max_length=120)
    type: str = Field(min_length=1, max_length=80)
    metric: str | None = None
    rule_category: str | None = None
    required: bool = True


class WorkcellInputSchema(StrictModel):
    path: str = "schemas/input.schema.json"


class WorkcellOutputSchema(StrictModel):
    path: str = "schemas/output.schema.json"


class WorkcellPolicy(StrictModel):
    enabled: bool = True
    tools: dict[str, bool] = Field(default_factory=dict)
    require_human_approval_for: list[str] = Field(default_factory=list)
    unsigned_allowed: bool | None = None


class WorkcellArtifactDefinition(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    artifact_type: str = Field(min_length=1, max_length=80)
    handler: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_]*$")
    filename: str = Field(min_length=1, max_length=255)
    derived_from_claims: list[str] = Field(default_factory=list)


class WorkcellEvaluationDefinition(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    description: str = ""
    input: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)


class WorkcellStep(StrictModel):
    id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_]*$")
    handler: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_]*$")
    depends_on: list[str] = Field(default_factory=list)
    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)
    condition: Literal["always", "input_present"] = "always"
    condition_input: str | None = None
    failure_behavior: Literal["stop", "continue"] = "stop"
    evidence_requirements: list[str] = Field(default_factory=list)
    approval_required: bool = False

    @model_validator(mode="after")
    def validate_condition(self):
        if self.condition == "input_present" and not self.condition_input:
            raise ValueError("input_present requires condition_input")
        return self


class WorkcellWorkflow(StrictModel):
    version: str = Field(min_length=1, max_length=40)
    steps: list[WorkcellStep] = Field(min_length=1)
    terminal_step: str


class WorkcellManifest(StrictModel):
    id: str = Field(min_length=2, max_length=100, pattern=r"^[a-z][a-z0-9-]*$")
    name: str = Field(min_length=2, max_length=160)
    version: str = Field(min_length=1, max_length=40)
    description: str = Field(min_length=1, max_length=2_000)
    platform_version: str = Field(min_length=1, max_length=80)
    task_classes: list[str] = Field(min_length=1)
    supported_execution_modes: list[Literal["FAST", "STANDARD", "DEEP"]] = Field(min_length=1)
    required_tools: list[str] = Field(default_factory=list)
    optional_tools: list[str] = Field(default_factory=list)
    risk_class: str = Field(min_length=1, max_length=80)
    entry_workflow: str = "workflow.yaml"
    input_schema: WorkcellInputSchema = Field(default_factory=WorkcellInputSchema)
    output_schema: WorkcellOutputSchema = Field(default_factory=WorkcellOutputSchema)
    enabled: bool = True

    @field_validator("required_tools", "optional_tools")
    @classmethod
    def unique_tools(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Tool names must be unique")
        return value

    @model_validator(mode="after")
    def no_overlapping_tools(self):
        overlap = set(self.required_tools) & set(self.optional_tools)
        if overlap:
            raise ValueError(f"Tools cannot be both required and optional: {sorted(overlap)}")
        return self


class WorkcellVersion(StrictModel):
    workcell_id: str
    version: str
    content_hash: str


class WorkcellValidationIssue(StrictModel):
    code: str
    message: str
    path: str | None = None


class WorkcellValidationResult(StrictModel):
    valid: bool
    status: WorkcellStatus
    workcell_id: str | None = None
    version: str | None = None
    content_hash: str | None = None
    trust_status: WorkcellTrustStatus = WorkcellTrustStatus.unsigned
    issues: list[WorkcellValidationIssue] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkcellDefinition(StrictModel):
    root: str
    manifest: WorkcellManifest
    workflow: WorkcellWorkflow
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    policy: WorkcellPolicy = Field(default_factory=WorkcellPolicy)
    evidence_requirements: list[WorkcellEvidenceRequirement] = Field(default_factory=list)
    rules: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[WorkcellArtifactDefinition] = Field(default_factory=list)
    evaluations: list[WorkcellEvaluationDefinition] = Field(default_factory=list)
    prompts: dict[str, str] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    content_hash: str
    files: dict[str, str]
    signature: dict[str, Any] | None = None


class WorkcellExecutionState(BaseModel):
    workcell_id: str
    workcell_version: str
    workcell_hash: str
    workflow_version: str
    current_step: str | None = None
    completed_steps: list[str] = Field(default_factory=list)
    failed_steps: list[str] = Field(default_factory=list)
    step_inputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    step_outputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    approvals: list[dict[str, Any]] = Field(default_factory=list)


class WorkcellCatalogEntry(BaseModel):
    id: str
    name: str
    version: str
    description: str
    task_classes: list[str]
    required_tools: list[str]
    status: WorkcellStatus
    trust_status: WorkcellTrustStatus
    content_hash: str | None = None
    validation: WorkcellValidationResult
