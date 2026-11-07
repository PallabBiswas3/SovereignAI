"""Check a provisioned Compose deployment; fail explicitly on unavailable checks.

Default checks append only a synthetic ControlPlane audit event. --restart-check
creates temporary marker files and recreates backend/ControlPlane containers.
Never run that option while operators have active tasks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import ssl
import subprocess
import time
import urllib.request
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default="deploy/.env")
    parser.add_argument("--file", action="append", help="Compose files in order; defaults to compose.yaml")
    parser.add_argument("--project-name", help="Explicit project, e.g. an isolated recovery/validation installation")
    parser.add_argument("--url", default="https://localhost:8443")
    parser.add_argument("--ca", type=Path, default=ROOT / "deploy/local-root.crt")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--restart-check", action="store_true")
    args = parser.parse_args()
    compose = ["docker", "compose", "--env-file", args.env_file]
    for file in args.file or ["compose.yaml"]:
        compose += ["-f", file]
    if args.project_name:
        compose += ["--project-name", args.project_name]

    def run(argv, **kwargs):
        return subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                              check=kwargs.pop("check", True), timeout=kwargs.pop("timeout", 60), **kwargs)

    def execute(service, source):
        return run(compose + ["exec", "-T", service, "python", "-"], input=source)

    checks = {}
    try:
        run(compose + ["config", "--quiet"])
        checks["compose_configuration"] = True
        run(["docker", "info"], timeout=20)
        checks["docker_engine"] = True
        probe = run(compose + ["exec", "-T", "backend", "python", "/app/backend/scripts/docker_probe.py"]
                    + (["--inference"] if args.inference else []), timeout=480, check=False)
        result = json.loads(probe.stdout)
        checks.update(result["checks"])
        if not args.url.startswith("https://"):
            raise ValueError("An HTTPS URL and trusted CA are required")
        context = ssl.create_default_context(cafile=str(args.ca))
        for name, path in (("browser_frontend", "/"), ("browser_api", "/health")):
            with urllib.request.urlopen(args.url.rstrip("/") + path, context=context, timeout=30) as response:
                checks[name] = response.status == 200
        if args.restart_check:
            # Markers prove writes survive replacement, not just process restart.
            # Paths are fixed; only the randomly named files created here are removed.
            token = "docker-smoke-" + uuid4().hex
            locations = {"backend": ["/app/backend/data", "/app/workspace/evidence_capsules",
                                      "/app/knowledge_base", "/app/organizations"],
                         "controlplane": ["/app/Controlplane.ai/results"]}
            try:
                for service, paths in locations.items():
                    execute(service, "from pathlib import Path\n" +
                            f"for root in {paths!r}:\n p=Path(root)/{token!r}; p.write_text({token!r})\n")
                run(compose + ["up", "-d", "--force-recreate", "--no-deps", "--no-build", "--pull", "never", "backend", "controlplane"], timeout=180)
                deadline = time.monotonic() + 180
                while True:
                    try:
                        run(compose + ["exec", "-T", "backend", "python", "-c",
                            "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=10)"], timeout=15)
                        break
                    except subprocess.CalledProcessError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Recreated backend did not become ready")
                        time.sleep(3)
                for service, paths in locations.items():
                    execute(service, "from pathlib import Path\n" +
                            f"for root in {paths!r}:\n assert (Path(root)/{token!r}).read_text()=={token!r}\n")
                audit_id = result["audit_event_id"]
                execute("controlplane", "from pathlib import Path\n" +
                        f"assert {audit_id!r} in Path('/app/Controlplane.ai/results/controlplane/audit.jsonl').read_text()\n")
                checks["evidence_and_audit_survive_recreation"] = True
            finally:
                for service, paths in locations.items():
                    try:
                        execute(service, "from pathlib import Path\n" +
                                f"for root in {paths!r}:\n (Path(root)/{token!r}).unlink(missing_ok=True)\n")
                    except (OSError, subprocess.SubprocessError):
                        checks["temporary_marker_cleanup"] = False
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        # Do not print captured application output: it may contain sensitive data.
        checks["completed"] = False
        print(json.dumps({"passed": False, "checks": checks, "error_type": type(exc).__name__,
                          "hint": "Check engine, provisioning, service logs, URL and CA; no unavailable check is a pass."}, indent=2))
        return 1
    print(json.dumps({"passed": all(checks.values()), "checks": checks,
                      "scope": "Synthetic checks; no consequential tool or plant action executed"}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
