"""Initialize file-mounted credentials without overwriting existing secrets.

No services are started, models downloaded or existing data migrated.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import time

ROOT = Path(__file__).resolve().parents[1]


def initialize(directory: Path) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    paths = [directory / name for name in (
        "graphrag_token", "postgres_password", "graph_db_password",
        "graph_rest_uri", "graph_jwt_secret", "graph_service_jwt",
    )]
    existing = [path.exists() for path in paths]
    if any(existing):
        if all(existing):
            return []
        raise ValueError("Partial secret set found; restore the complete set instead of rotating database credentials")
    values = {name: secrets.token_urlsafe(48) for name in (
        "graphrag_token", "postgres_password", "graph_db_password", "graph_jwt_secret")}
    values["graph_rest_uri"] = f"postgres://graph_auth:{values['graph_db_password']}@graph-db:5432/evidence"
    def encode(value):
        return base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode()).rstrip(b"=")
    issued = int(time.time())
    message = encode({"alg": "HS256", "typ": "JWT"}) + b"." + encode({
        "role": "graph_service", "iss": "sovereign-local", "iat": issued,
        "exp": issued + 365 * 24 * 3600,
    })
    signature = hmac.new(values["graph_jwt_secret"].encode(), message, hashlib.sha256).digest()
    values["graph_service_jwt"] = (message + b"." + base64.urlsafe_b64encode(signature).rstrip(b"=")).decode()
    for path in paths:
        # The parent directory is owner-only on POSIX. Readable mounted files
        # allow non-root container users to consume Compose's bind secrets.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(values[path.name])
    return [path.name for path in paths]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        created = initialize(ROOT / "deploy" / "secrets")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    (ROOT / "deploy" / "models").mkdir(exist_ok=True)
    print("Created credential files: " + ", ".join(created) if created else "Existing credentials preserved")
    print("Provision local model files and identity, then run scripts/preflight_production.py. See deploy/README.md.")
