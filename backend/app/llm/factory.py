from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.llm.base import LocalModelProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.vllm_provider import VLLMProvider
from app.router.schemas import ModelDefinition


@dataclass(frozen=True, slots=True)
class LocalProviderSelection:
    provider: LocalModelProvider
    model: str
    provider_name: str
    endpoint: str


def configured_local_provider(
    settings: Settings | None = None,
    *,
    model_definition: ModelDefinition | None = None,
    ollama_model: str | None = None,
    execution_mode: str = "STANDARD",
    priority: int = 50,
) -> LocalProviderSelection:
    settings = settings or get_settings()
    provider_name = settings.llm_provider.strip().lower()
    role = model_definition.role if model_definition else "GENERAL"
    memory_requirement = model_definition.memory_requirement if model_definition else "medium"
    if provider_name == "vllm":
        provider = VLLMProvider(
            settings.vllm_url,
            api_key=settings.vllm_api_key,
            allow_fallback=settings.allow_deterministic_fallback,
            enable_thinking=settings.vllm_enable_thinking,
            max_tokens=settings.vllm_max_tokens,
            timeout_seconds=settings.model_generation_timeout_seconds,
            role=role,
            memory_requirement=memory_requirement,
            execution_mode=execution_mode,
            priority=priority,
        )
        return LocalProviderSelection(provider, settings.vllm_model, "vllm", provider.endpoint)
    if provider_name == "ollama":
        endpoint = model_definition.endpoint if model_definition else settings.ollama_url
        model = ollama_model or (model_definition.model_tag if model_definition else "qwen3:4b-instruct")
        provider = OllamaProvider(
            endpoint,
            settings.allow_deterministic_fallback,
            role=role,
            memory_requirement=memory_requirement,
            execution_mode=execution_mode,
            priority=priority,
            timeout_seconds=settings.model_generation_timeout_seconds,
        )
        return LocalProviderSelection(provider, model, "ollama", provider.endpoint)
    raise ValueError(f"Unsupported local model provider: {settings.llm_provider!r}")
