from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


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
    allowed_origins: list[str] = ["http://127.0.0.1:3000", "http://localhost:3000"]
    graphrag_service_token_file: Path | None = None
    readiness_timeout_seconds: float = Field(default=3.0, gt=0, le=15)
    workspace_root: Path = PROJECT_DIR / "workspace"
    knowledge_root: Path = PROJECT_DIR / "knowledge_base"
    ollama_url: str = "http://127.0.0.1:11434"
    llm_provider: str = "ollama"
    vllm_url: str = "http://127.0.0.1:8001/v1"
    vllm_model: str = "Qwen/Qwen3-0.6B"
    vllm_api_key: str = "local"
    vllm_enable_thinking: bool = False
    vllm_max_tokens: int = 1280
    allow_deterministic_fallback: bool = True
    max_upload_mb: int = 25
    voice_model_path: Path = PROJECT_DIR / "models" / "faster-whisper-tiny.en"
    voice_language: str = Field(default="en", pattern=r"^[a-z]{2,3}$")
    voice_cpu_threads: int = Field(default=2, ge=1, le=16)
    voice_max_duration_seconds: int = Field(default=30, ge=1, le=120)
    voice_max_upload_bytes: int = Field(default=8 * 1024 * 1024, ge=1024, le=16 * 1024 * 1024)
    voice_upload_timeout_seconds: float = Field(default=15.0, gt=0, le=60)
    voice_timeout_seconds: float = Field(default=90.0, gt=0, le=600)
    embedding_provider: str = "semantic"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_local_files_only: bool = True
    embedding_allow_hash_fallback: bool = True
    max_gpu_model_jobs: int = 1
    max_cpu_jobs: int = 2
    model_idle_timeout_seconds: int = 300
    model_keep_alive: str | None = None
    model_generation_timeout_seconds: float = 600.0
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
    graphrag_service_token: str = ""
    controlplane_url: str = "http://127.0.0.1:8100"
    diagnostics_url: str = "http://127.0.0.1:8200"
    integration_timeout_seconds: float = 180.0
    integration_max_retries: int = 2
    integration_retry_backoff_seconds: float = 0.25
    integration_retry_backoff_max_seconds: float = 2.0
    integration_fail_closed: bool = True


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.graphrag_service_token_file:
        settings.graphrag_service_token = settings.graphrag_service_token_file.read_text(encoding="utf-8").strip()
    validate_runtime_settings(settings)
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    settings.knowledge_root.mkdir(parents=True, exist_ok=True)
    settings.organizations_root.mkdir(parents=True, exist_ok=True)
    settings.organization_reports_root.mkdir(parents=True, exist_ok=True)
    (BACKEND_DIR / "data").mkdir(parents=True, exist_ok=True)
    return settings


def validate_runtime_settings(settings: Settings) -> None:
    """Reject unsafe production configuration before opening the API."""
    if settings.environment.lower() != "production":
        return
    if settings.auth_mode.lower() == "disabled":
        raise RuntimeError("SOVEREIGN_AUTH_MODE=disabled is forbidden in production")
    if settings.auth_mode.lower() != "local":
        raise RuntimeError("Production requires the supported local identity provider")
    if not settings.auth_cookie_secure:
        raise RuntimeError("Production requires secure session cookies and HTTPS")
    if settings.demo_org_enabled:
        raise RuntimeError("Production forbids synthetic demo account seeding")
    if not settings.integration_fail_closed or settings.allow_deterministic_fallback:
        raise RuntimeError("Production requires fail-closed integrations and disabled deterministic fallback")
    if len(settings.graphrag_service_token) < 32:
        raise RuntimeError("Production requires a GraphRAG service credential of at least 32 characters")
    from urllib.parse import urlparse
    for origin in settings.allowed_origins:
        try:
            parsed = urlparse(origin)
            parsed.port  # Validate malformed/out-of-range ports as well.
        except ValueError as exc:
            raise RuntimeError("Production CORS origins must be explicit HTTPS origins") from exc
        if (parsed.scheme != "https" or not parsed.hostname or "*" in origin or parsed.username or parsed.password
                or parsed.path or parsed.query or parsed.fragment):
            raise RuntimeError("Production CORS origins must be explicit HTTPS origins")
    if not settings.allowed_origins:
        raise RuntimeError("Production requires at least one explicit HTTPS origin")
