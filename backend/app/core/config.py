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
    max_gpu_model_jobs: int = 1
    max_cpu_jobs: int = 2
    model_idle_timeout_seconds: int = 300
    model_keep_alive: str | None = None
    model_generation_timeout_seconds: float = 300.0
    cache_enabled: bool = True
    cache_default_ttl_seconds: int | None = None
    hybrid_dense_top_k: int = 30
    hybrid_sparse_top_k: int = 30
    hybrid_fusion_candidate_limit: int = 50
    hybrid_rrf_k: int = 60
    hybrid_rerank_top_k: int = 10
    hybrid_final_context_k: int = 5
    dense_retriever_version: str = "cosine-v2"
    bm25_index_version: str = "bm25-v1"
    fusion_strategy_version: str = "rrf-v1"
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_local_files_only: bool = True
    reranker_version: str = "cross-encoder-v1"
    context_max_fraction_of_window: float = 0.60
    context_output_reserve_tokens: int = 1024
    context_max_evidence_chunks: int = 8
    context_max_evidence_tokens: int = 3000
    context_near_duplicate_threshold: float = 0.90
    max_retrieval_subqueries: int = 4


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    settings.knowledge_root.mkdir(parents=True, exist_ok=True)
    (BACKEND_DIR / "data").mkdir(parents=True, exist_ok=True)
    return settings
