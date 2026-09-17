import json

import pytest

from tsdiag.evaluation.split_manifest import (
    DevelopmentSplitManifest,
    SplitUnit,
    load_split_manifest,
    save_split_manifest,
)
from tsdiag.evaluation.wind_calibration import build_wind_split_manifest


def test_unit_boundary_rejects_event_overlap():
    manifest = DevelopmentSplitManifest(
        dataset_id="dataset-v1",
        development_pool_id="development",
        evaluation_pool_id="frozen-test",
        leakage_boundary="unit",
        unit_kind="event",
        group_kind=None,
        development_units=(SplitUnit("7"),),
        evaluation_units=(SplitUnit("7"),),
    )
    with pytest.raises(ValueError, match="overlap"):
        manifest.validate()


def test_group_boundary_rejects_shared_asset_and_missing_identity():
    shared = DevelopmentSplitManifest(
        dataset_id="dataset-v1",
        development_pool_id="development",
        evaluation_pool_id="frozen-test",
        leakage_boundary="group",
        unit_kind="event",
        group_kind="asset",
        development_units=(SplitUnit("1", ("asset-a",)),),
        evaluation_units=(SplitUnit("2", ("asset-a",)),),
    )
    with pytest.raises(ValueError, match="share asset"):
        shared.validate()

    missing = DevelopmentSplitManifest(
        dataset_id="dataset-v1",
        development_pool_id="development",
        evaluation_pool_id="frozen-test",
        leakage_boundary="group",
        unit_kind="event",
        group_kind="asset",
        development_units=(SplitUnit("1"),),
        evaluation_units=(SplitUnit("2", ("asset-b",)),),
    )
    with pytest.raises(ValueError, match="identity is missing"):
        missing.validate()


def test_temporal_boundary_requires_development_before_evaluation():
    valid = DevelopmentSplitManifest(
        dataset_id="dataset-v1",
        development_pool_id="development",
        evaluation_pool_id="frozen-test",
        leakage_boundary="temporal",
        unit_kind="event",
        group_kind="asset",
        development_units=(
            SplitUnit("1", ("asset-a",), "2024-01-01", "2024-01-31"),
        ),
        evaluation_units=(
            SplitUnit("2", ("asset-a",), "2024-02-01", "2024-02-28"),
        ),
    )
    valid.validate()

    invalid = DevelopmentSplitManifest(
        **{
            **valid.__dict__,
            "evaluation_units": (
                SplitUnit("2", ("asset-a",), "2024-01-15", "2024-02-28"),
            ),
        }
    )
    with pytest.raises(ValueError, match="not strictly earlier"):
        invalid.validate()


def test_manifest_round_trip_verifies_checksum(tmp_path):
    manifest = DevelopmentSplitManifest(
        dataset_id="dataset-v1",
        development_pool_id="development",
        evaluation_pool_id="frozen-test",
        leakage_boundary="unit",
        unit_kind="event",
        group_kind=None,
        development_units=(SplitUnit("1"),),
        evaluation_units=(SplitUnit("2"),),
    )
    path = save_split_manifest(manifest, tmp_path / "split.json")
    assert load_split_manifest(path).sha256() == manifest.sha256()

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["development_pool_id"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_split_manifest(path)


def test_wind_asset_boundary_uses_farm_scoped_asset_identity():
    manifest = build_wind_split_manifest(
        [{"event_id": 1, "wind_farm": "A", "asset_ids": [3]}],
        [{"event_id": 2, "wind_farm": "B", "asset_ids": [3]}],
        dataset_id="care-v6",
        development_pool_id="development",
        evaluation_pool_id="frozen-test",
        leakage_boundary="asset",
    )
    assert manifest.development_units[0].group_ids == ("farm:A/asset:3",)
    assert manifest.evaluation_units[0].group_ids == ("farm:B/asset:3",)
