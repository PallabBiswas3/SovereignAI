from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tsdiag.datasets.bearing_lenze import LENZE_ARCHIVE_SIZE, download_lenze_mb


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume the official Lenze-MB download")
    parser.add_argument("--data-dir", default="data/bearing_lenze")
    args = parser.parse_args()
    paths = download_lenze_mb(args.data_dir)
    archive = paths["archive"]
    print(json.dumps({
        "archive": str(archive),
        "bytes": archive.stat().st_size,
        "expected_bytes": LENZE_ARCHIVE_SIZE,
        "extracted_data": str(paths["data"]),
    }, indent=2))


if __name__ == "__main__":
    main()
