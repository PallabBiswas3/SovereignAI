"""Read-only deployment checks. A failed check always produces exit status 1."""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.monitoring.model_manifest import load_lock, verify_directory
SECRET_NAMES = ("graphrag_token", "postgres_password", "graph_db_password", "graph_rest_uri", "graph_jwt_secret", "graph_service_jwt")
MODEL_NAMES = ("nli-deberta-v3-small", "all-MiniLM-L6-v2", "ms-marco-MiniLM-L-6-v2")


def secret_checks(directory: Path) -> dict[str, bool]:
    result = {}
    values = {}
    for name in SECRET_NAMES:
        try:
            values[name] = (directory / name).read_text(encoding="utf-8").strip()
            result[name] = len(values[name]) >= 32
        except OSError:
            result[name] = False
    if not all(result.values()):
        return result
    result["database_credential_matches"] = values["graph_rest_uri"] == f"postgres://graph_auth:{values['graph_db_password']}@graph-db:5432/evidence"
    try:
        head, body, signature = values["graph_service_jwt"].split(".")
        expected = hmac.new(values["graph_jwt_secret"].encode(), f"{head}.{body}".encode(), hashlib.sha256).digest()
        actual = base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4))
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        result["graph_jwt_valid"] = hmac.compare_digest(expected, actual) and payload.get("role") == "graph_service" and payload.get("exp", 0) > time.time() + 86400
    except (ValueError, TypeError, KeyError):
        result["graph_jwt_valid"] = False
    return result


def model_files_present(directory: Path) -> bool:
    return (directory / "config.json").is_file() and (directory / "tokenizer_config.json").is_file() and any(
        (directory / name).is_file() for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"))


def run_check(command: list[str]) -> bool:
    try:
        return subprocess.run(command, cwd=ROOT, capture_output=True, timeout=30, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", action="store_true", help="Check source configuration only; does not establish runtime readiness")
    parser.add_argument("--url", help="Also check the deployed HTTPS /ready endpoint")
    parser.add_argument("--ca", type=Path, help="Trusted deployment CA; TLS verification is never disabled")
    args = parser.parse_args()
    checks = {"python_3_11": sys.version_info >= (3, 11)}
    try:
        import yaml
        profile = yaml.safe_load((ROOT / "deploy/compose.production.yml").read_text(encoding="utf-8"))
        services = profile["services"]
        checks["complete_service_profile"] = {"backend", "frontend", "graphrag", "controlplane", "diagnostics", "ollama", "gateway", "graph-db", "graph-rest"} <= services.keys()
        checks["single_ingress"] = all("ports" not in config for name, config in services.items() if name != "gateway")
        checks["isolated_networks"] = all(config.get("internal") is True for config in profile["networks"].values())
        environment = services["backend"]["environment"]
        checks["production_auth"] = environment.get("SOVEREIGN_ENVIRONMENT") == "production" and environment.get("SOVEREIGN_AUTH_MODE") == "local" and environment.get("SOVEREIGN_AUTH_COOKIE_SECURE") == "true"
        checks["no_host_docker_socket"] = "docker.sock" not in json.dumps(profile)
    except (ImportError, OSError, ValueError, KeyError, TypeError):
        checks["profile_configuration"] = False
    if not args.static:
        # Lower bound for image extraction, build intermediates and the separate
        # model volume. This is a provisioning gate, not a runtime accuracy claim.
        checks["host_provisioning_space_10gib"] = shutil.disk_usage(ROOT).free >= 10 * 1024 ** 3
        checks.update({f"secret:{name}": ok for name, ok in secret_checks(ROOT / "deploy/secrets").items()})
        try:
            locked = load_lock(ROOT / "deploy/model-lock.json")
            checks.update({f"model_integrity:{name}": verify_directory(ROOT / "deploy/models" / name, model)["valid"]
                           for name, model in locked.items()})
        except (OSError, ValueError, KeyError, TypeError):
            checks["model_lock"] = False
        checks["docker_engine"] = bool(shutil.which("docker")) and run_check(["docker", "info"])
        checks["compose_configuration"] = run_check(["docker", "compose", "-f", "deploy/compose.production.yml", "config", "--quiet"])
    if args.url:
        if not args.url.startswith("https://"):
            checks["deployed_readiness"] = False
        else:
            try:
                context = ssl.create_default_context(cafile=str(args.ca) if args.ca else None)
                with urllib.request.urlopen(args.url.rstrip("/") + "/ready", context=context, timeout=15) as response:
                    checks["deployed_readiness"] = response.status == 200 and json.load(response).get("ready") is True
            except (OSError, ValueError, urllib.error.URLError):
                checks["deployed_readiness"] = False
    print(json.dumps({"passed": all(checks.values()), "mode": "static" if args.static else "deployment-preflight", "checks": checks,
                      "capacity_scope": "Filesystem containing this checkout; also verify Docker's backing disk if it differs.",
                      "scope": "Preflight does not establish authenticated workflow acceptance or model accuracy."}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
