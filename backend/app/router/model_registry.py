from __future__ import annotations

from pathlib import Path
import os

import yaml

from app.router.schemas import ModelDefinition


class ModelRegistry:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self._models = self._load()

    def _load(self) -> dict[str, ModelDefinition]:
        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        definitions: dict[str, ModelDefinition] = {}
        for model_id, values in raw.get("models", {}).items():
            # Explicit deployment overrides apply only to Ollama entries. Defaults
            # and other providers retain their existing registry behavior.
            values = dict(values)
            if values.get("provider") == "ollama":
                endpoint = os.environ.get("SOVEREIGN_OLLAMA_URL")
                model = os.environ.get(f"SOVEREIGN_OLLAMA_{model_id.upper()}_MODEL")
                if endpoint:
                    values["endpoint"] = endpoint
                if model:
                    values["model_tag"] = model
            try:
                definitions[model_id] = ModelDefinition(id=model_id, **values)
            except Exception as exc:
                raise ValueError(f"Invalid model configuration for '{model_id}': {exc}") from exc
        if not definitions:
            raise ValueError("Model registry contains no models")
        return definitions

    def all(self) -> list[ModelDefinition]:
        return list(self._models.values())

    def get(self, model_id: str) -> ModelDefinition:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model override: {model_id}") from exc
