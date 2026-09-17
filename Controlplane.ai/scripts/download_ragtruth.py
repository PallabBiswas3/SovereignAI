#!/usr/bin/env python3
"""
Fetch RAGTruth into data/raw/ragtruth/.

RAGTruth is distributed on GitHub (not HuggingFace), as:
    github.com/ParticleMedia/RAGTruth
with the actual data at dataset/response.jsonl and dataset/source_info.jsonl
inside that repo. This script does a shallow clone and copies just the
`dataset/` folder out, so we don't keep the repo's git history around.

Usage:
    python scripts/download_ragtruth.py
    python scripts/download_ragtruth.py --dest data/raw/ragtruth
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/ParticleMedia/RAGTruth.git"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        default="data/raw/ragtruth",
        help="Destination directory for response.jsonl / source_info.jsonl",
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    if (dest / "response.jsonl").exists() and (dest / "source_info.jsonl").exists():
        print(f"Already present at {dest}, skipping download.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        print(f"Cloning {REPO_URL} (shallow) ...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, tmp],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            sys.exit(1)

        src_dataset_dir = Path(tmp) / "dataset"
        if not src_dataset_dir.exists():
            print(f"Expected {src_dataset_dir} not found after clone.", file=sys.stderr)
            sys.exit(1)

        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src_dataset_dir, dest)

    n_resp = sum(1 for _ in open(dest / "response.jsonl", encoding="utf-8"))
    n_src = sum(1 for _ in open(dest / "source_info.jsonl", encoding="utf-8"))
    print(f"Done. {dest}/response.jsonl: {n_resp} lines, "
          f"{dest}/source_info.jsonl: {n_src} lines.")


if __name__ == "__main__":
    main()
