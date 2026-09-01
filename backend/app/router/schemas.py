from __future__ import annotations

from pydantic import BaseModel, Field
from enum import Enum


class TaskProfile(BaseModel):
    task_type: str
    coding_requirement: float = Field(ge=0, le=1)
    reasoning_requirement: float = Field(ge=0, le=1)
    vision_requirement: float = Field(ge=0, le=1)
    document_requirement: float = Field(ge=0, le=1)
    summarization_requirement: float = Field(ge=0, le=1)
    latency_priority: float = Field(ge=0, le=1)
    context_length_required: int = Field(ge=1)


class ModelAvailability(str, Enum):
    ready = "READY"
    model_not_installed = "MODEL_NOT_INSTALLED"
    ollama_unavailable = "OLLAMA_UNAVAILABLE"
    configuration_error = "CONFIGURATION_ERROR"
    unknown = "UNKNOWN"


class ModelDefinition(BaseModel):
    id: str
    role: str
    display_name: str
    provider: str
    model_tag: str
    endpoint: str
    capabilities: dict[str, float]
    context_length: int
    estimated_latency: str = "medium"
    memory_requirement: str = "medium"
    supports_text: bool = True
    supports_images: bool = False
    supports_tools: bool = False
    availability: ModelAvailability = ModelAvailability.unknown


class ModelRuntimeStatus(BaseModel):
    id: str
    role: str
    display_name: str
    model_tag: str
    endpoint: str
    availability: ModelAvailability
    installed: bool
    detail: str
    capabilities: list[str] = Field(default_factory=list)
    lifecycle_state: str = "UNKNOWN"
    warm_status: str = "unknown"
    memory_usage_mb: float | None = None
    runtime_metrics: dict[str, object] = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    selected_model: str
    model_id: str
    confidence: float
    reason: str
    task_profile: TaskProfile
    manual_override: bool = False
    scores: dict[str, float]
