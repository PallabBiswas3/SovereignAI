from __future__ import annotations

import asyncio
import ipaddress
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import yaml
from sqlalchemy.orm import Session

from app.core.database import NetworkEventRecord


def _docker_daemon_available() -> bool:
    """Probe Docker without relying on asyncio subprocess support on Windows."""
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2,
            check=False,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.SubprocessError, NotImplementedError):
        return False
    return completed.returncode == 0


class LocalNetworkPolicy:
    ALLOWED_SERVICE_NAMES = {"backend", "frontend", "qdrant", "ollama", "sandbox", "ocr"}

    @classmethod
    def is_local_url(cls, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        if host in {"localhost", *cls.ALLOWED_SERVICE_NAMES}:
            return True
        try:
            address = ipaddress.ip_address(host)
            return address.is_loopback
        except ValueError:
            return False

    @classmethod
    def require_local(cls, url: str) -> None:
        if not cls.is_local_url(url):
            raise ValueError(f"External network destination blocked by sovereignty policy: {url}")


class NetworkMonitor:
    def __init__(self, session: Session | None = None) -> None:
        self.session = session

    def record_attempt(self, url: str, component: str, allowed: bool) -> None:
        if not self.session:
            return
        self.session.add(NetworkEventRecord(
            id=str(uuid4()), destination=url, component=component,
            allowed=allowed, created_at=datetime.now(timezone.utc),
        ))
        self.session.commit()

    def counts(self) -> dict[str, int]:
        if not self.session:
            return {"external_attempts": 0, "allowed_local_requests": 0}
        external = self.session.query(NetworkEventRecord).filter_by(allowed=False).count()
        local = self.session.query(NetworkEventRecord).filter_by(allowed=True).count()
        return {"external_attempts": external, "allowed_local_requests": local}


class AirGapVerifier:
    def verify_model_config(self, path: Path) -> dict[str, object]:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        endpoints = [values.get("endpoint", "") for values in raw.get("models", {}).values()]
        violations = [endpoint for endpoint in endpoints if not LocalNetworkPolicy.is_local_url(str(endpoint))]
        return {"passed": not violations, "checked_endpoints": endpoints, "violations": violations}


async def local_service_status(ollama_url: str) -> list[dict[str, object]]:
    services: list[dict[str, object]] = [
        {"name": "Backend", "endpoint": "127.0.0.1:8000", "status": "active"},
        {"name": "Frontend", "endpoint": "127.0.0.1:3000", "status": "configured"},
        {"name": "Vector database", "endpoint": "embedded SQLite", "status": "active"},
        {"name": "OCR", "endpoint": "local Tesseract", "status": "available" if shutil.which("tesseract") else "unavailable"},
    ]
    try:
        LocalNetworkPolicy.require_local(ollama_url)
        async with httpx.AsyncClient(timeout=1.5, follow_redirects=False) as client:
            response = await client.get(f"{ollama_url.rstrip('/')}/api/tags")
            response.raise_for_status()
        llm_status = "active"
    except (httpx.HTTPError, ValueError):
        llm_status = "unavailable"
    services.append({"name": "LLM", "endpoint": ollama_url, "status": llm_status})
    if not shutil.which("docker"):
        docker_status = "unavailable"
    else:
        try:
            available = await asyncio.wait_for(asyncio.to_thread(_docker_daemon_available), timeout=3)
            docker_status = "available" if available else "unavailable"
        except (RuntimeError, asyncio.TimeoutError):
            docker_status = "unavailable"
    services.append({"name": "Sandbox", "endpoint": "Docker isolated", "status": docker_status})
    return services
