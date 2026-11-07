"""Synthetic checks inside the backend; no protected tool is executed.

The ControlPlane denial check intentionally appends a synthetic audit event.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from pathlib import Path
import sys
from urllib.parse import urlparse
import urllib.error
import urllib.request

# This script is also invoked by absolute path inside the container. In that
# mode Python adds scripts/, rather than backend/, to its import search path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.governance.action_guard import ActionGuard
from app.governance.policy_engine import GovernanceDecision
from app.router.model_registry import ModelRegistry
from app.llm.factory import configured_local_provider


def request(url, payload=None, timeout=15):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", action="store_true", help="Also generate a short answer on the configured runtime and a local embedding")
    args = parser.parse_args()
    settings = get_settings()
    checks = {}
    for name, url in (("backend", "http://127.0.0.1:8000/health"),
                      ("readiness", "http://127.0.0.1:8000/ready"),
                      ("controlplane", settings.controlplane_url + "/health"),
                      ("diagnostics", settings.diagnostics_url + "/health"),
                      ("graphrag", settings.graphrag_url + "/health")):
        try:
            code, body = request(url)
            checks[name] = code == 200 and (body.get("ready") is True if name == "readiness" else body.get("status") == "ok")
        except (OSError, ValueError):
            checks[name] = False
    # Browser clients must authenticate even when they can reach the gateway.
    checks["anonymous_api_denied"] = request("http://127.0.0.1:8000/api/auth/me")[0] == 401
    checks["anonymous_graph_denied"] = request(settings.graphrag_url + "/api/integration/retrieve", {})[0] == 401
    guard = ActionGuard(settings.tools_config)
    checks["unknown_tool_denied"] = guard.evaluate("docker_smoke_unregistered").decision == GovernanceDecision.block
    checks["approval_cannot_enable_shell"] = guard.evaluate("execute_shell", approved=True).decision == GovernanceDecision.block
    code, report = request(settings.controlplane_url + "/v1/precheck", {
        "profile": "internal_assistant", "prompt": "Synthetic smoke test: api_key=" + "X" * 32,
    }, timeout=45)
    checks["controlplane_secret_denied"] = code == 200 and report.get("decision", {}).get("action") == "block"
    checks["controlplane_audited"] = bool(report.get("audit_event_id"))
    # Read the real database without modifying application evidence or tasks.
    database = Path(urlparse(settings.database_url).path)
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        checks["sqlite_integrity"] = db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    code, tags = request(settings.ollama_url + "/api/tags")
    general = next(m for m in ModelRegistry(settings.models_config).all() if m.role == "GENERAL")
    checks["local_model_installed"] = code == 200 and any(
        item.get("name") in {general.model_tag, general.model_tag + ":latest"} for item in tags.get("models", []))
    if args.inference:
        if settings.llm_provider.strip().lower() in {"remote_vllm", "hf_space"}:
            try:
                runtime = configured_local_provider(settings)
                result = asyncio.run(runtime.provider.generate("Synthetic smoke test. Reply with OK.", runtime.model))
                checks["remote_generation"] = bool(result.text.strip()) and not result.fallback
            except (OSError, ValueError, RuntimeError):
                checks["remote_generation"] = False
        else:
            code, result = request(settings.ollama_url + "/api/generate", {
                "model": general.model_tag, "prompt": "Reply with OK.", "stream": False, "think": False,
                "options": {"num_predict": 8, "num_ctx": 2048}, "keep_alive": "0s",
            }, timeout=300)
            checks["local_generation"] = code == 200 and bool(result.get("response", "").strip())
        import os
        code, result = request(settings.ollama_url + "/api/embed", {
            "model": os.environ.get("SOVEREIGN_SMOKE_EMBEDDING_MODEL", "nomic-embed-text"),
            "input": "Synthetic local evidence probe.", "keep_alive": "0s",
        }, timeout=120)
        checks["local_embedding"] = code == 200 and bool(result.get("embeddings", [[]])[0])
    print(json.dumps({"passed": all(checks.values()), "checks": checks,
                      "audit_event_id": report.get("audit_event_id"),
                      "scope": "Synthetic deployment checks; not plant or workflow acceptance"}))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
