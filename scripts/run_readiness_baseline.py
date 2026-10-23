"""Run maintained checks with isolated storage and record exact source identity."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def source_identity():
    def git(*args):
        return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT)
    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
              for path in paths if path and (ROOT / path).is_file()}
    return {"commit": git("rev-parse", "HEAD").decode().strip(), "source_file_sha256": hashes,
            "branch": git("branch", "--show-current").decode().strip(),
            "working_tree": git("status", "--porcelain").decode(),
            "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD")).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-local-controlplane", action="store_true", help="Include ignored research tests when available; not a clean-checkout requirement")
    args = parser.parse_args()
    out = ROOT / "local_testing" / ("readiness-baseline-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True)
    env = os.environ.copy()
    env.update(SOVEREIGN_ENVIRONMENT="development", SOVEREIGN_AUTH_MODE="disabled",
               SOVEREIGN_EMBEDDING_PROVIDER="hash", SOVEREIGN_CACHE_ENABLED="false",
               SOVEREIGN_DATABASE_URL=f"sqlite:///{(out / 'baseline.db').as_posix()}",
               PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in ("backend", "Controlplane.ai/src", "Time-Series-Diagnostic-Agent/src", "local_testing")))
    npm = shutil.which("npm.cmd") or shutil.which("npm") or "npm"
    jobs = [("python", ROOT, [sys.executable, "-m", "pytest", f"--junitxml={out / 'python.xml'}"]),
            ("frontend-tests", ROOT / "frontend", ["node", "--test", "agent-tools-tests/components.test.cjs", "voice-tests/recorder.test.cjs", "tests/task-stream.test.cjs"]),
            ("frontend-typecheck", ROOT / "frontend", [npm, "run", "typecheck"]),
            ("graphrag-build", ROOT / "Graph-RAG/server", [npm, "run", "build"]),
            ("graphrag-auth", ROOT / "Graph-RAG/server", ["node", "--test", "security/serviceAuth.test.cjs"])]
    if args.include_local_controlplane:
        jobs.append(("controlplane-local", ROOT / "Controlplane.ai", [sys.executable, "-m", "pytest", "../local_testing/Controlplane.ai/tests", "tests", "-o", "addopts=", "-q", f"--junitxml={out / 'controlplane.xml'}"]))
    manifest = {"source": source_identity(), "started_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Software regression checks; no live deployment or detector accuracy claim", "jobs": []}
    for name, cwd, command in jobs:
        with (out / f"{name}.log").open("w", encoding="utf-8") as stream:
            try:
                result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
                code = result.returncode
            except (OSError, subprocess.SubprocessError):
                code = -1
        manifest["jobs"].append({"name": name, "cwd": str(cwd.relative_to(ROOT)), "command": command, "exit_code": code})
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"{name}: {'passed' if code == 0 else 'failed'}")
    print(f"Evidence: {out}")
    return int(any(job["exit_code"] != 0 for job in manifest["jobs"]))


if __name__ == "__main__":
    raise SystemExit(main())
