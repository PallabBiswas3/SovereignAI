from __future__ import annotations

from app.core.config import Settings
from app.llm.base import LocalModelProvider
from app.llm.llama_cpp_provider import LlamaCppProvider
from app.llm.ollama_provider import OllamaProvider
from app.resources.backend_selection import BackendSelectionStore


def create_local_model_provider(
    settings: Settings,
    *,
    model: str,
    ollama_endpoint: str,
    role: str = "GENERAL",
    memory_requirement: str = "medium",
    execution_mode: str = "STANDARD",
    priority: int = 50,
) -> LocalModelProvider:
    """Create the validated local inference backend for a logical model.

    Ollama remains the fail-safe default. llama.cpp is used only when the machine-local benchmark
    has persisted an explicit selection for this model.
    """

    selection = (
        BackendSelectionStore(settings.model_backend_selection_path).get(model)
        if settings.model_backend_selection_enabled
        else None
    )
    if selection is not None and selection.backend == "llama_cpp":
        return LlamaCppProvider(
            selection.endpoint or settings.llama_cpp_url,
            settings.allow_deterministic_fallback,
            role=role,
            memory_requirement=memory_requirement,
            execution_mode=execution_mode,
            priority=priority,
            server_context_size=settings.llama_cpp_context_size,
        )
    return OllamaProvider(
        ollama_endpoint,
        settings.allow_deterministic_fallback,
        role=role,
        memory_requirement=memory_requirement,
        execution_mode=execution_mode,
        priority=priority,
    )
