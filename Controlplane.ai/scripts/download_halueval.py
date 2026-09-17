#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

FILES = {
    "qa_data.json": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json",
    "general_data.json": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/general_data.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the HaluEval files used in Phase 4")
    parser.add_argument("--output", default="data/raw/halueval")
    parser.add_argument("--include-general", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    names = ["qa_data.json"]
    if args.include_general:
        names.append("general_data.json")

    for name in names:
        destination = output / name
        if destination.exists():
            print(f"Already exists: {destination}")
            continue
        print(f"Downloading {name}...")
        urlretrieve(FILES[name], destination)
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
