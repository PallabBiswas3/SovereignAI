"""Record a bounded verification command and its output in the local evidence log.

Commands passed here must never contain or print credentials. No environment
values are recorded. Evidence is local-only and excluded from container builds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--cwd", type=Path, default=ROOT)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    directory = ROOT / "local_testing/deployment-remediation-20261023"
    directory.mkdir(parents=True, exist_ok=True)
    if not command or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.label):
        parser.error("A safe label and a command are required")
    started = datetime.now(timezone.utc).isoformat()
    output = directory / f"{args.label}.log"
    if output.exists():
        parser.error("Evidence labels must be unique; prior output is never overwritten")
    beginning = time.monotonic()
    status = "finished"
    with output.open("wb") as stream:
        try:
            result = subprocess.run(command, cwd=args.cwd, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=args.timeout, check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code, status = 124, "timeout"
        except OSError:
            code, status = 127, "could_not_start"
    record = {"label": args.label, "started_utc": started, "seconds": round(time.monotonic() - beginning, 3),
              "command": command, "cwd": str(args.cwd), "exit_code": code, "status": status,
              "output": str(output.relative_to(ROOT)), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    with (directory / "commands.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record))
    print(output.read_text(encoding="utf-8", errors="replace")[-7000:])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
