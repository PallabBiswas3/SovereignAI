from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Runtime settings. Every default is local and air-gap safe."""

    model_config = SettingsConfigDict(env_prefix="SOVEREIGN_", env_file=".env", extra="ignore")

    app_name: str = "SovereignAI Workbench"
    environment: str = "development"
    database_url: str = f"sqlite:///{(BACKEND_DIR / 'data' / 'sovereign.db').as_posix()}"
    models_config: Path = PROJECT_DIR / "config" / "models.yaml"
    policies_config: Path = PROJECT_DIR / "config" / "policies.yaml"
    tools_config: Path = PROJECT_DIR / "config" / "tools.yaml"
    workspace_root: Path = PROJECT_DIR / "workspace"
    knowledge_root: Path = PROJECT_DIR / "knowledge_base"
    ollama_url: str = "http://127.0.0.1:11434"
    allow_deterministic_fallback: bool = True
    max_upload_mb: int = 25
    embedding_provider: str = "semantic"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_local_files_only: bool = True
    embedding_allow_hash_fallback: bool = True


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    settings.knowledge_root.mkdir(parents=True, exist_ok=True)
    (BACKEND_DIR / "data").mkdir(parents=True, exist_ok=True)
    return settings
