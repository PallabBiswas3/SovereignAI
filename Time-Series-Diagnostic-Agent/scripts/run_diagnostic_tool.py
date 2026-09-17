from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tsdiag.integrations import diagnose_tool


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the tsdiag JSON tool boundary.")
    parser.add_argument("--request", help="Request JSON file; stdin is used when omitted")
    args = parser.parse_args()
    payload = Path(args.request).read_text(encoding="utf-8") if args.request else sys.stdin.read()
    print(json.dumps(diagnose_tool(payload), indent=2))


if __name__ == "__main__":
    main()
