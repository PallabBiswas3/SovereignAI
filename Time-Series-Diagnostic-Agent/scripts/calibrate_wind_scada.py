from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tsdiag.benchmarks.wind_scada import _aligned_sensor_matrices
from tsdiag.datasets.wind_care import (
    CARE_ZENODO_RECORD,
    care_normal_mask,
    load_care_event,
    load_care_event_asset_ids,
    load_care_event_info,
)
from tsdiag.evaluation.split_manifest import save_split_manifest
from tsdiag.evaluation.wind_calibration import (
    DEFAULT_WIND_CALIBRATION_PATH,
    build_wind_split_manifest,
    calibrate_wind_residual_config,
    save_wind_calibration_result,
)


def _ids(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _frozen_ids(path: str | Path) -> list[int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [int(row["event_id"]) for row in payload["events"]]


def _time_text(value) -> str | None:
    if value is None or str(value) in {"NaT", "nan"}:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _event_metadata(data_dir: str | Path, event_ids: list[int], *, include_assets: bool):
    info = load_care_event_info(data_dir)
    records = []
    for event_id in event_ids:
        matches = info[info["event_id"].astype(int) == int(event_id)]
        if matches.empty:
            raise KeyError(f"CARE event_id={event_id} not found")
        row = matches.iloc[0]
        records.append({
            "event_id": event_id,
            "wind_farm": str(row["wind_farm"]),
            "asset_ids": (
                load_care_event_asset_ids(data_dir, event_id) if include_assets else ()
            ),
            "event_start": _time_text(row.get("event_start")),
            "event_end": _time_text(row.get("event_end")),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate Wind-SCADA event FAR on a disjoint healthy development pool"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--development-event-ids", required=True)
    parser.add_argument("--development-pool-id", required=True)
    parser.add_argument("--evaluation-pool-id", default="care-v6-frozen-95")
    parser.add_argument("--dataset-id", default=f"care-v6:zenodo-{CARE_ZENODO_RECORD}")
    parser.add_argument(
        "--leakage-boundary",
        choices=("event", "asset", "temporal"),
        required=True,
        help=(
            "Explicit disjointness claim. Asset mode fails on missing/shared asset IDs; "
            "temporal mode additionally requires development intervals to precede evaluation."
        ),
    )
    frozen = parser.add_mutually_exclusive_group(required=True)
    frozen.add_argument(
        "--frozen-report",
        help="Existing frozen benchmark report whose event IDs must not overlap development",
    )
    frozen.add_argument(
        "--frozen-event-ids",
        help="Comma-separated evaluation IDs predeclared before any evaluation results are inspected",
    )
    parser.add_argument("--target-event-far", type=float, default=0.08)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--search-report",
        default="outputs/wind_calibration/wind_calibration_search.json",
    )
    parser.add_argument(
        "--manifest-output",
        default="outputs/wind_calibration/development_split_manifest.json",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Validate and write the leakage manifest without running grid search",
    )
    parser.add_argument(
        "--freeze-config",
        default=str(DEFAULT_WIND_CALIBRATION_PATH),
        help="Written only when the requested target FAR is met",
    )
    args = parser.parse_args()

    development_ids = _ids(args.development_event_ids)
    frozen_event_ids = (
        _frozen_ids(args.frozen_report)
        if args.frozen_report
        else _ids(args.frozen_event_ids)
    )
    include_assets = args.leakage_boundary in {"asset", "temporal"}
    development_metadata = _event_metadata(
        args.data_dir, development_ids, include_assets=include_assets
    )
    evaluation_metadata = _event_metadata(
        args.data_dir, frozen_event_ids, include_assets=include_assets
    )
    split_manifest = build_wind_split_manifest(
        development_metadata,
        evaluation_metadata,
        dataset_id=args.dataset_id,
        development_pool_id=args.development_pool_id,
        evaluation_pool_id=args.evaluation_pool_id,
        leakage_boundary=args.leakage_boundary,
    )
    manifest_path = save_split_manifest(split_manifest, args.manifest_output)
    if args.manifest_only:
        print(json.dumps({
            "manifest": str(manifest_path),
            "manifest_sha256": split_manifest.sha256(),
            "leakage_boundary": split_manifest.leakage_boundary,
            "development_units": len(split_manifest.development_units),
            "evaluation_units": len(split_manifest.evaluation_units),
        }, indent=2))
        return

    events = []
    metadata_by_id = {int(row["event_id"]): row for row in development_metadata}
    for event_id in development_ids:
        event_data = load_care_event(args.data_dir, event_id, statistics=("avg",))
        train, _, channel_names, _, _ = _aligned_sensor_matrices(event_data)
        events.append({
            **metadata_by_id[event_id],
            "event_id": event_id,
            "train_matrix": train,
            "healthy_train_mask": np.asarray(care_normal_mask(event_data.train), dtype=bool),
            "channel_names": channel_names,
        })

    result = calibrate_wind_residual_config(
        events,
        split_manifest=split_manifest,
        target_event_far=args.target_event_far,
        holdout_fraction=args.holdout_fraction,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    search_path = Path(args.search_report)
    search_path.parent.mkdir(parents=True, exist_ok=True)
    search_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    frozen_path = None
    if result.met_target:
        frozen_path = save_wind_calibration_result(result, args.freeze_config)
    print(json.dumps({
        "met_target": result.met_target,
        "target_event_far": result.target_event_far,
        "achieved_event_far": result.achieved_event_far,
        "achieved_event_far_ci": result.achieved_event_far_ci,
        "achieved_sample_far": result.achieved_sample_far,
        "n_events_used": result.n_events_used,
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": result.split_manifest_sha256,
        "search_report": str(search_path),
        "frozen_config": None if frozen_path is None else str(frozen_path),
    }, indent=2))


if __name__ == "__main__":
    main()
