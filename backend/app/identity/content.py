from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ContentIdentityService:
    """SHA-256 identities using UTF-8 and canonical compact JSON."""

    @staticmethod
    def hash_bytes(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    def hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def canonical_json(value: Any) -> bytes:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")

    def hash_json(self, value: Any) -> str:
        return self.hash_bytes(self.canonical_json(value))

    def hash_directory_manifest(self, files: dict[str, str]) -> str:
        normalized = [
            {"path": path.replace("\\", "/"), "sha256": digest}
            for path, digest in sorted(files.items(), key=lambda item: item[0].replace("\\", "/"))
        ]
        return self.hash_json(normalized)

    def directory_manifest(self, root: Path, paths: list[Path]) -> dict[str, str]:
        resolved_root = root.resolve()
        manifest: dict[str, str] = {}
        for path in sorted(paths, key=lambda item: item.as_posix()):
            resolved = path.resolve()
            if resolved_root != resolved and resolved_root not in resolved.parents:
                raise ValueError("File escapes identity root")
            manifest[resolved.relative_to(resolved_root).as_posix()] = self.hash_file(resolved)
        return manifest
