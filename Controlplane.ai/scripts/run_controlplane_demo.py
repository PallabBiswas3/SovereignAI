#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlplane import ControlPlane, Interaction
from controlplane.verification import build_adaptive_verification_from_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one interaction through ControlPlane.ai")
    parser.add_argument("--profile", default="customer_support")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--context")
    parser.add_argument("--context-file")
    parser.add_argument("--consequential", action="store_true")
    parser.add_argument("--no-audit", action="store_true")
    args = parser.parse_args()

    context = args.context
    if args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8")
    checker = ControlPlane(
        audit_enabled=not args.no_audit,
        verification_service=build_adaptive_verification_from_env(),
    )
    report = checker.check(
        Interaction(
            profile=args.profile,
            prompt=args.prompt,
            response=args.response,
            context=context,
            consequential=args.consequential,
        )
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
