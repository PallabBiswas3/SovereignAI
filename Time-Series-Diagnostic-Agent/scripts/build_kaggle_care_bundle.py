#!/usr/bin/env python3
"""Build the code-only Kaggle bundle for the CARE v6 benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = "tsdiag-care-kaggle"
INCLUDE_PATHS = (
    Path("pyproject.toml"),
    Path("README.md"),
    Path("docs/PROJECT_DIRECTION.md"),
    Path("src"),
    Path("kaggle"),
    Path("scripts/run_wind_scada_benchmark.py"),
)
SKIP_NAMES = {"__pycache__", ".pytest_cache"}


def _files() -> list[Path]:
    files: list[Path] = []
    for relative in INCLUDE_PATHS:
        source = ROOT / relative
        if source.is_file():
            files.append(source)
        elif source.is_dir():
            files.extend(
                path for path in source.rglob("*")
                if path.is_file()
                and not any(part in SKIP_NAMES for part in path.relative_to(ROOT).parts)
                and path.suffix not in {".pyc", ".pyo"}
            )
        else:
            raise FileNotFoundError(source)
    return sorted(set(files))


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ("git", "-c", f"safe.directory={ROOT.as_posix()}", *args),
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "tsdiag-care-kaggle.zip")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    files = _files()
    records = []
    tree_digest = sha256()
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for source in files:
            relative = source.relative_to(ROOT).as_posix()
            content = source.read_bytes()
            digest = sha256(content).hexdigest()
            records.append({"path": relative, "sha256": digest, "bytes": len(content)})
            tree_digest.update(relative.encode("utf-8"))
            tree_digest.update(bytes.fromhex(digest))
            archive.writestr(f"{BUNDLE_ROOT}/{relative}", content)

        status = _git_value("status", "--porcelain")
        manifest = {
            "bundle_schema": "tsdiag.kaggle-care-bundle",
            "bundle_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_git_sha": _git_value("rev-parse", "HEAD"),
            "source_worktree_dirty": bool(status),
            "source_tree_sha256": tree_digest.hexdigest(),
            "contains_care_dataset": False,
            "file_count": len(records),
            "files": records,
        }
        archive.writestr(
            f"{BUNDLE_ROOT}/BUNDLE_MANIFEST.json",
            json.dumps(manifest, indent=2).encode("utf-8"),
        )

    print(json.dumps({
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output.read_bytes()).hexdigest(),
        "source_tree_sha256": tree_digest.hexdigest(),
        "file_count": len(records),
    }, indent=2))


if __name__ == "__main__":
    main()
