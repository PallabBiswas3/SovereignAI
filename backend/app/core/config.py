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
    access_config: Path = PROJECT_DIR / "config" / "access.yaml"
    workcells_root: Path = PROJECT_DIR / "workcells"
    organizations_root: Path = PROJECT_DIR / "organizations"
    organization_reports_root: Path = PROJECT_DIR / "workspace" / "onboarding_reports"
    unsigned_workcells_allowed: bool = True
    capsules_root: Path = PROJECT_DIR / "workspace" / "evidence_capsules"
    unsigned_capsules_allowed: bool = True
    auth_mode: str = "disabled"
    demo_org_enabled: bool = False
    auth_session_seconds: int = 8 * 60 * 60
    auth_cookie_name: str = "sovereign_session"
    auth_csrf_cookie_name: str = "sovereign_csrf"
    auth_cookie_secure: bool = False
    workspace_root: Path = PROJECT_DIR / "workspace"
    knowledge_root: Path = PROJECT_DIR / "knowledge_base"
    ollama_url: str = "http://127.0.0.1:11434"
    llama_cpp_url: str = "http://127.0.0.1:8080"
    llama_cpp_context_size: int = 4096
    llama_cpp_kv_cache_type_k: str = "f16"
    llama_cpp_kv_cache_type_v: str = "f16"
    model_backend_selection_enabled: bool = True
    model_backend_selection_path: Path = BACKEND_DIR / "data" / "backend_selection.json"
    model_backend_min_speedup: float = 1.05
    model_kv_cache_optimization_enabled: bool = True
    model_kv_cache_optimization_path: Path = BACKEND_DIR / "data" / "kv_cache_optimization.json"
    model_kv_cache_min_quality: float = 0.95
    model_kv_cache_max_tps_regression: float = 0.05
    model_kv_cache_min_ram_saving_mb: float = 128.0
    allow_deterministic_fallback: bool = True
    max_upload_mb: int = 25
    embedding_provider: str = "semantic"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_local_files_only: bool = True
    embedding_allow_hash_fallback: bool = True
    # Deprecated compatibility setting. CPU-only systems should use model_max_concurrent_jobs.
    max_gpu_model_jobs: int = 1
    max_cpu_jobs: int = 2
    model_max_concurrent_jobs: int | None = 1
    model_ram_admission_enabled: bool = True
    model_ram_reserve_mb: float = 1536.0
    model_ram_reserve_fraction: float = 0.15
    model_footprints_path: Path = BACKEND_DIR / "data" / "model_footprints.json"
    model_footprint_safety_multiplier: float = 1.10
    model_adaptive_residency_enabled: bool = True
    model_cpu_tuning_enabled: bool = True
    model_cpu_tuning_path: Path = BACKEND_DIR / "data" / "cpu_tuning.json"
    model_adaptive_context_enabled: bool = True
    model_context_buckets: str = "2048,4096,8192,16384"
    model_context_minimum: int = 2048
    model_context_bytes_per_token: float = 3.0
    model_context_prompt_margin_tokens: int = 128
    model_optimization_enabled: bool = True
    model_optimization_path: Path = BACKEND_DIR / "data" / "model_optimization.json"
    model_optimization_min_quality: float = 0.90
    # CPU-laptop default: preserve fast conversational follow-ups without holding a 3+ GB model
    # resident for many minutes when system RAM is already under heavy pressure.
    model_idle_timeout_seconds: int = 60
    model_keep_alive: str | None = None
    model_generation_timeout_seconds: float = 600.0
    model_prewarm_on_startup: bool = False
    model_prewarm_ids: str = "general"
    model_prewarm_keep_alive: str = "60s"
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
    telemetry_default_freshness_seconds: int = 300
    telemetry_expired_seconds: int = 86400
    telemetry_scenario: str = "PUMP_102_DEGRADING"
    graphrag_url: str = "http://127.0.0.1:3100"
    controlplane_url: str = "http://127.0.0.1:8100"
    diagnostics_url: str = "http://127.0.0.1:8200"
    integration_timeout_seconds: float = 180.0
    integration_fail_closed: bool = True


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.environment.lower() == "production" and settings.auth_mode.lower() == "disabled":
        raise RuntimeError("SOVEREIGN_AUTH_MODE=disabled is forbidden in production")
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    settings.knowledge_root.mkdir(parents=True, exist_ok=True)
    settings.organizations_root.mkdir(parents=True, exist_ok=True)
    settings.organization_reports_root.mkdir(parents=True, exist_ok=True)
    settings.model_footprints_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_cpu_tuning_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_optimization_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_backend_selection_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_kv_cache_optimization_path.parent.mkdir(parents=True, exist_ok=True)
    (BACKEND_DIR / "data").mkdir(parents=True, exist_ok=True)
    return settings
