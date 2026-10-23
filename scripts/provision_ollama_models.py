"""Verify cached locked model blobs, provision a named volume and test recreation.

Copies only model manifests/blobs, never host Ollama identity keys. The temporary
daemon has no network or published ports. Existing conflicting tags are refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def docker(*args, **kwargs):
    print(json.dumps({"command": ["docker", *args]}), flush=True)
    return subprocess.run(["docker", *args], check=True, **kwargs)


def check_existing_tags(name, lock):
    """Only an explicitly absent regular manifest permits provisioning."""
    for model in lock["models"].values():
        path = "/root/.ollama/models/" + model["manifest_path"]
        result = subprocess.run([
            "docker", "exec", name, "sh", "-c",
            'if [ -L "$1" ]; then exit 1; elif [ -f "$1" ]; then cat "$1"; '
            'elif [ -e "$1" ]; then exit 1; else exit 44; fi', "sh", path,
        ], capture_output=True)
        if result.returncode == 44:
            continue
        if result.returncode != 0:
            raise RuntimeError("Cannot inspect existing volume tag; no overwrite permitted")
        if hashlib.sha256(result.stdout).hexdigest() != model["manifest_sha256"]:
            raise RuntimeError("Existing volume tag conflicts with the model lock; no overwrite performed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".ollama/models")
    parser.add_argument("--volume", required=True)
    parser.add_argument("--image", default="ollama/ollama:0.33.2")
    parser.add_argument("--storage-only", action="store_true", help="Verify persisted model bytes without claiming that the Ollama daemon ran")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]+", args.volume):
        raise SystemExit("Explicit valid volume name required")
    lock_path = ROOT / "deploy/ollama-model-lock.json"
    lock = json.loads(lock_path.read_text())
    unique_blobs = {digest: size for model in lock["models"].values() for digest, size in model["blobs"].items()}
    minimum = sum(unique_blobs.values()) + 1024 ** 3
    available = shutil.disk_usage(ROOT).free
    if available < minimum:
        print(json.dumps({"provisioned": False, "reason": "insufficient_host_staging_space", "free_bytes": available, "minimum_bytes": minimum}), flush=True)
        return 2
    # Use exactly the same verifier on host and inside the persistent volume.
    import runpy
    verify = runpy.run_path(str(ROOT / "tests/deployment/ollama_volume_verify.py"))["verify"]
    checked = verify(args.cache, lock)
    print(json.dumps({"host_cache": checked}), flush=True)
    if not checked["valid"]:
        return 1
    docker("volume", "create", args.volume, stdout=subprocess.DEVNULL)
    name = "sovereign-model-check-" + uuid4().hex[:12]
    started = False
    def start():
        image = "python:3.11-slim-bookworm" if args.storage_only else args.image
        command = ["python", "-c", "import time; time.sleep(600)"] if args.storage_only else []
        docker("run", "-d", "--network", "none", "--name", name, "--volume", f"{args.volume}:/root/.ollama", image, *command, stdout=subprocess.DEVNULL)
        # Synchronous creation prevents docker cp racing the container entrypoint.
        docker("exec", name, "mkdir", "-p", "/root/.ollama/models", stdout=subprocess.DEVNULL)
        if args.storage_only:
            return
        for _ in range(30):
            result = subprocess.run(["docker", "exec", name, "ollama", "list"], capture_output=True)
            if result.returncode == 0:
                return
            time.sleep(1)
        raise RuntimeError("Isolated Ollama daemon did not start")
    def volume_check():
        docker("run", "--rm", "--network", f"container:{name}",
               "--mount", f"type=volume,source={args.volume},target=/models,readonly",
               "--mount", f"type=bind,source={lock_path},target=/lock.json,readonly",
               "--mount", f"type=bind,source={ROOT / 'tests/deployment/ollama_volume_verify.py'},target=/verify.py,readonly",
               "python:3.11-slim-bookworm", "python", "/verify.py", *([] if args.storage_only else ["--api"]))
    try:
        started = True
        start()
        check_existing_tags(name, lock)
        files = {model["manifest_path"] for model in lock["models"].values()}
        files.update("blobs/" + digest.replace(":", "-") for model in lock["models"].values() for digest in model["blobs"])
        process = subprocess.Popen(["docker", "cp", "-", f"{name}:/root/.ollama/models"], stdin=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdin, mode="w|") as archive:
                for filename in sorted(files):
                    archive.add(args.cache / filename, arcname=filename, recursive=False)
            process.stdin.close()
            if process.wait(timeout=300) != 0:
                raise RuntimeError("Model volume copy failed")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        volume_check()
        docker("rm", "-f", name, stdout=subprocess.DEVNULL)
        started = False
        started = True
        start()
        volume_check()
        print(json.dumps({"volume": args.volume, "recreated_container": True, "daemon_verified": not args.storage_only, "persistent_model_integrity": True}), flush=True)
    finally:
        if started:
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
