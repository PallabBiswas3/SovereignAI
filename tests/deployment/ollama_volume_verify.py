"""Read-only model-volume checks, optionally with an isolated Ollama daemon."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def verify(root, lock):
    errors = []
    blobs = {}
    for tag, model in lock["models"].items():
        path = root / model["manifest_path"]
        try:
            if hashlib.sha256(path.read_bytes()).hexdigest() != model["manifest_sha256"]:
                errors.append(f"{tag}:manifest_hash")
        except OSError:
            errors.append(f"{tag}:manifest_missing")
        blobs.update(model["blobs"])
    for digest, size in blobs.items():
        path = root / "blobs" / digest.replace(":", "-")
        try:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if path.stat().st_size != size or actual != digest.removeprefix("sha256:"):
                errors.append(f"{digest}:blob_hash")
        except OSError:
            errors.append(f"{digest}:blob_missing")
    return {"valid": not errors, "models": len(lock["models"]), "blobs": len(blobs), "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/models"))
    parser.add_argument("--lock", type=Path, default=Path("/lock.json"))
    parser.add_argument("--api", action="store_true")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    result = verify(args.root, lock)
    if args.api:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=15) as response:
            values = json.load(response)["models"]
        installed = {item["name"]: item["digest"].removeprefix("sha256:") for item in values}
        result["api_digests_match"] = all(installed.get(tag) == model["manifest_sha256"] for tag, model in lock["models"].items())
        result["valid"] = result["valid"] and result["api_digests_match"]
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["valid"] else 1)
