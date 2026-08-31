"""Active network-isolation proof intended to run inside the deployed backend container."""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


def probe(url: str, timeout: float = 4.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return True, f"HTTP {response.status}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-url", default=os.getenv("SOVEREIGN_OLLAMA_URL", "http://ollama:11434"))
    parser.add_argument("--require-ollama", action="store_true")
    args = parser.parse_args()
    external_ok, external_detail = probe("https://example.com/")
    ollama_ok, ollama_detail = probe(f"{args.ollama_url.rstrip('/')}/api/tags")
    report = {
        "external_egress_blocked": not external_ok,
        "external_probe": external_detail,
        "ollama_reachable": ollama_ok,
        "ollama_probe": ollama_detail,
        "passed": (not external_ok) and (ollama_ok or not args.require_ollama),
        "scope": "active probe from this process/network namespace",
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
