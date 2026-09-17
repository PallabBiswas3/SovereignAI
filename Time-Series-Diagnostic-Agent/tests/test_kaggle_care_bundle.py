from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
from zipfile import ZipFile

import pytest


MODULE_PATH = Path(__file__).parents[1] / "kaggle" / "run_care_kaggle.py"
SPEC = spec_from_file_location("tsdiag_kaggle_care", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
KAGGLE_RUNNER = module_from_spec(SPEC)
SPEC.loader.exec_module(KAGGLE_RUNNER)


def _care_archive(path: Path, event_ids=(1, 2, 3)) -> Path:
    with ZipFile(path, "w") as archive:
        for farm, event_id in zip(("A", "B", "C"), event_ids):
            prefix = f"Wind Farm {farm}"
            archive.writestr(
                f"{prefix}/event_info.csv",
                f"event_id;asset_id;event_label\n{event_id};0;normal\n",
            )
            archive.writestr(
                f"{prefix}/feature_description.csv",
                "sensor_name;statistics_type\nwind_speed;average\n",
            )
            archive.writestr(f"{prefix}/datasets/.keep", "")
    return path


def test_kaggle_runner_discovers_one_complete_care_archive(tmp_path):
    archive = _care_archive(tmp_path / "CARE_To_Compare.zip")

    assert KAGGLE_RUNNER.discover_care_path(tmp_path) == archive


def test_kaggle_runner_rejects_incomplete_care_archive(tmp_path):
    (tmp_path / "CARE_To_Compare.zip").write_bytes(b"partial download")

    with pytest.raises(ValueError, match="incomplete"):
        KAGGLE_RUNNER.discover_care_path(tmp_path)


def test_kaggle_smoke_selection_respects_wind_farm(tmp_path):
    archive = _care_archive(tmp_path / "CARE_To_Compare.zip", event_ids=(10, 20, 30))

    assert KAGGLE_RUNNER._selected_event_ids(archive, "smoke", 1, "B") == [20]
    assert KAGGLE_RUNNER._selected_event_ids(archive, "full", 1, "B") is None


def test_kaggle_runner_loads_source_bundle_identity(tmp_path):
    manifest = {
        "bundle_schema": "tsdiag.kaggle-care-bundle",
        "bundle_version": "1.0",
        "source_tree_sha256": "a" * 64,
    }
    (tmp_path / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert KAGGLE_RUNNER.load_bundle_manifest(tmp_path) == manifest


def test_kaggle_runner_rejects_unknown_bundle_manifest(tmp_path):
    (tmp_path / "BUNDLE_MANIFEST.json").write_text(
        json.dumps({"bundle_schema": "unknown"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unexpected bundle manifest schema"):
        KAGGLE_RUNNER.load_bundle_manifest(tmp_path)
