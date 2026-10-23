"""Verify local model bytes against a reviewed, immutable Hub file inventory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def load_lock(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not value.get("models"):
        raise ValueError("Invalid model lock")
    for model in value["models"].values():
        if not re.fullmatch(r"[0-9a-f]{40}", model.get("revision", "")) or not model.get("files"):
            raise ValueError("Model lock must name an immutable commit and its files")
        for name, expected in model["files"].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise ValueError("Invalid model file path")
            if not isinstance(expected.get("size"), int) or expected["size"] < 1:
                raise ValueError("Invalid locked file size")
            digest = expected.get("sha256") or expected.get("git_blob_sha1")
            if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", digest or ""):
                raise ValueError("Missing model file hash")
    return value["models"]


def verify_directory(directory: Path, model: dict) -> dict:
    errors, hashes = [], {}
    try:
        provenance = json.loads((directory / "sovereign-snapshot.json").read_text(encoding="utf-8"))
        if provenance.get("repository") != model["repository"] or provenance.get("revision") != model["revision"]:
            errors.append("snapshot_revision_mismatch")
    except (OSError, ValueError):
        errors.append("snapshot_provenance_missing")
    for name, expected in model["files"].items():
        path = directory / name
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                errors.append(f"{name}:external_reference")
                continue
            size = path.stat().st_size
            if size != expected["size"]:
                errors.append(f"{name}:size_mismatch")
                continue
            sha = hashlib.sha256()
            blob = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    sha.update(chunk)
                    blob.update(chunk)
            hashes[name] = sha.hexdigest()
            actual = sha.hexdigest() if expected.get("sha256") else blob.hexdigest()
            if actual != (expected.get("sha256") or expected.get("git_blob_sha1")):
                errors.append(f"{name}:hash_mismatch")
        except OSError:
            errors.append(f"{name}:missing_or_unreadable")
    return {"valid": not errors, "revision": model["revision"], "errors": errors, "sha256": hashes}
