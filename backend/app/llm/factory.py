from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.llm.base import LocalModelProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.vllm_provider import VLLMProvider


@dataclass(frozen=True, slots=True)
class LocalProviderSelection:
    provider: LocalModelProvider
    model: str
    provider_name: str
    endpoint: str


def configured_local_provider(
    settings: Settings | None = None,
    *,
    ollama_model: str = "qwen3:4b-instruct",
) -> LocalProviderSelection:
    settings = settings or get_settings()
    provider_name = settings.llm_provider.strip().lower()
    if provider_name == "vllm":
        provider = VLLMProvider(
            settings.vllm_url,
            api_key=settings.vllm_api_key,
            allow_fallback=settings.allow_deterministic_fallback,
            enable_thinking=settings.vllm_enable_thinking,
            max_tokens=settings.vllm_max_tokens,
            timeout_seconds=settings.model_generation_timeout_seconds,
        )
        return LocalProviderSelection(provider, settings.vllm_model, "vllm", settings.vllm_url)
    if provider_name == "ollama":
        provider = OllamaProvider(
            settings.ollama_url,
            settings.allow_deterministic_fallback,
            role="GENERAL",
            memory_requirement="medium",
            execution_mode="STANDARD",
        )
        return LocalProviderSelection(provider, ollama_model, "ollama", settings.ollama_url)
    raise ValueError(f"Unsupported local model provider: {settings.llm_provider!r}")
