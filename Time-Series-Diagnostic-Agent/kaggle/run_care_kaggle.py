#!/usr/bin/env python3
"""Run the canonical CARE v6 benchmark inside a Kaggle notebook.

The tsdiag source bundle and the official CARE archive are expected to be
attached as separate Kaggle datasets. CARE is read directly from its ZIP; it
is not extracted into Kaggle's limited working storage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import md5
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Iterable
from zipfile import BadZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tsdiag.benchmarks.wind_scada_public import run_care_benchmark  # noqa: E402
from tsdiag.datasets.wind_care import (  # noqa: E402
    CARE_ARCHIVE_NAME,
    load_care_event_info,
    validate_care_layout,
)
from tsdiag.detectors.residual_changepoint import (  # noqa: E402
    DEFAULT_CUSUM_DRIFT,
    DEFAULT_CUSUM_HOLD_SAMPLES,
    DEFAULT_CUSUM_THRESHOLD,
)
from tsdiag.pipeline import PIPELINE_VERSION  # noqa: E402


OFFICIAL_CARE_MD5 = "2547b58c21ac8c242d13232860cf500c"
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def load_bundle_manifest(project_root: Path = PROJECT_ROOT) -> dict:
    path = project_root / "BUNDLE_MANIFEST.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("bundle_schema") != "tsdiag.kaggle-care-bundle":
        raise ValueError(f"unexpected bundle manifest schema in {path}")
    return payload


def _valid_care_path(path: Path) -> bool:
    try:
        status = validate_care_layout(path)
    except (BadZipFile, FileNotFoundError, OSError, ValueError):
        return False
    return bool(status) and all(status.values())


def discover_care_path(input_root: Path = KAGGLE_INPUT) -> Path:
    """Find one unambiguous official archive or extracted CARE root."""
    archives = sorted(input_root.rglob(CARE_ARCHIVE_NAME)) if input_root.exists() else []
    valid_archives = [path for path in archives if _valid_care_path(path)]
    if len(valid_archives) == 1:
        return valid_archives[0]
    if len(valid_archives) > 1:
        raise RuntimeError(
            "multiple CARE archives found; pass --care-path explicitly: "
            + ", ".join(str(path) for path in valid_archives)
        )

    roots: set[Path] = set()
    if input_root.exists():
        for info_file in input_root.rglob("event_info.csv"):
            if info_file.parent.name.startswith("Wind Farm "):
                roots.add(info_file.parent.parent)
    valid_roots = sorted(path for path in roots if _valid_care_path(path))
    if len(valid_roots) == 1:
        return valid_roots[0]
    if len(valid_roots) > 1:
        raise RuntimeError(
            "multiple extracted CARE datasets found; pass --care-path explicitly: "
            + ", ".join(str(path) for path in valid_roots)
        )
    if archives:
        raise ValueError(
            f"Found {CARE_ARCHIVE_NAME}, but it is incomplete or does not contain all three Wind Farm layouts: "
            + ", ".join(str(path) for path in archives)
        )
    raise FileNotFoundError(
        f"Could not find {CARE_ARCHIVE_NAME} or an extracted CARE v6 layout under {input_root}. "
        "Attach the official CARE archive as a Kaggle Dataset."
    )


def file_md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _selected_event_ids(
    care_path: Path,
    mode: str,
    smoke_events: int,
    wind_farm: str | None,
) -> list[int] | None:
    if mode == "full":
        return None
    if smoke_events < 1:
        raise ValueError("--smoke-events must be at least 1")
    info = load_care_event_info(care_path, wind_farm=wind_farm)
    return sorted(info["event_id"].astype(int).unique().tolist())[:smoke_events]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CARE v6 reproducibly on Kaggle")
    parser.add_argument("--care-path", help="Official CARE ZIP or extracted CARE root; auto-detected when omitted")
    parser.add_argument("--output-dir", default=str(KAGGLE_WORKING / "care_results"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--smoke-events", type=int, default=3)
    parser.add_argument("--wind-farm", choices=("A", "B", "C"))
    parser.add_argument("--verify-md5", action="store_true", help="Verify the 5.5 GB official ZIP before running")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    care_path = Path(args.care_path).resolve() if args.care_path else discover_care_path()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_md5 = None
    if args.verify_md5:
        if not care_path.is_file():
            raise ValueError("--verify-md5 requires the official CARE ZIP, not an extracted directory")
        archive_md5 = file_md5(care_path)
        if archive_md5 != OFFICIAL_CARE_MD5:
            raise RuntimeError(
                f"CARE archive checksum mismatch: expected {OFFICIAL_CARE_MD5}, got {archive_md5}"
            )

    bundle_manifest = load_bundle_manifest()
    artifact_checksums: dict[str, str] = {}
    source_tree_hash = bundle_manifest.get("source_tree_sha256")
    if source_tree_hash:
        artifact_checksums["tsdiag_source_tree_sha256"] = str(source_tree_hash)
    if archive_md5:
        artifact_checksums["care_archive_md5"] = archive_md5

    event_ids = _selected_event_ids(care_path, args.mode, args.smoke_events, args.wind_farm)
    expected_events = len(event_ids) if event_ids is not None else len(
        load_care_event_info(care_path, wind_farm=args.wind_farm)
    )
    result = run_care_benchmark(
        care_path,
        event_ids=event_ids,
        wind_farm=args.wind_farm,
        cusum_hold_samples=DEFAULT_CUSUM_HOLD_SAMPLES,
        artifact_checksums=artifact_checksums,
        output_dir=output_dir,
    )

    successful = int(result["summary"]["successful_events"])
    failed = int(result["summary"]["failed_events"])
    provenance_count = len(result.get("run_provenance", []))
    checks = {
        "expected_event_count": expected_events,
        "processed_event_count": successful + failed,
        "successful_event_count": successful,
        "failed_event_count": failed,
        "provenance_count": provenance_count,
        "all_events_accounted_for": successful + failed == expected_events,
        "provenance_aligned": provenance_count == successful,
        "canonical_cusum_defaults": (
            result["configuration"]["cusum_drift"] == DEFAULT_CUSUM_DRIFT
            and result["configuration"]["cusum_threshold"] == DEFAULT_CUSUM_THRESHOLD
            and result["configuration"]["cusum_hold_samples"] == DEFAULT_CUSUM_HOLD_SAMPLES
        ),
    }
    manifest = {
        "run_schema": "tsdiag.kaggle-care-run",
        "run_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "pipeline_version": PIPELINE_VERSION,
        "care_path": str(care_path),
        "care_archive_md5": archive_md5,
        "official_care_md5": OFFICIAL_CARE_MD5,
        "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": _versions((
            "numpy", "pandas", "scipy", "scikit-learn", "joblib", "statsmodels", "PyWavelets"
        )),
        "source_bundle": {
            key: bundle_manifest.get(key)
            for key in (
                "bundle_schema", "bundle_version", "created_at_utc", "source_git_sha",
                "source_worktree_dirty", "source_tree_sha256", "file_count",
            )
            if bundle_manifest.get(key) is not None
        },
        "artifact_checksums": artifact_checksums,
        "checks": checks,
        "summary": result["summary"],
    }
    (output_dir / "kaggle_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))

    if not all((checks["all_events_accounted_for"], checks["provenance_aligned"], checks["canonical_cusum_defaults"])):
        raise SystemExit("CARE run integrity checks failed")
    if failed and not args.allow_failures:
        raise SystemExit(f"CARE run completed with {failed} failed event(s)")


if __name__ == "__main__":
    main()
