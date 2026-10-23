import hashlib
import json

import pytest

from app.monitoring.model_manifest import load_lock, verify_directory


@pytest.fixture
def locked(tmp_path):
    config = b'{"model_type":"bert"}'
    weights = b"test-only-weight-bytes"
    (tmp_path / "config.json").write_bytes(config)
    (tmp_path / "model.safetensors").write_bytes(weights)
    model = {"repository": "test/model", "revision": "a" * 40, "files": {
        "config.json": {"size": len(config), "git_blob_sha1": hashlib.sha1(f"blob {len(config)}\0".encode() + config).hexdigest()},
        "model.safetensors": {"size": len(weights), "sha256": hashlib.sha256(weights).hexdigest()},
    }}
    (tmp_path / "sovereign-snapshot.json").write_text(json.dumps({"repository": model["repository"], "revision": model["revision"]}))
    return tmp_path, model


def test_locked_files_and_immutable_provenance_verify(locked):
    root, model = locked
    assert verify_directory(root, model)["valid"] is True


def test_same_size_weight_corruption_is_rejected(locked):
    root, model = locked
    data = (root / "model.safetensors").read_bytes()
    (root / "model.safetensors").write_bytes(b"X" + data[1:])
    assert "model.safetensors:hash_mismatch" in verify_directory(root, model)["errors"]


def test_git_blob_file_corruption_is_rejected(locked):
    root, model = locked
    data = (root / "config.json").read_bytes()
    (root / "config.json").write_bytes(b"X" + data[1:])
    assert "config.json:hash_mismatch" in verify_directory(root, model)["errors"]


def test_forged_snapshot_revision_is_rejected(locked):
    root, model = locked
    (root / "sovereign-snapshot.json").write_text(json.dumps({"repository": model["repository"], "revision": "b" * 40}))
    assert "snapshot_revision_mismatch" in verify_directory(root, model)["errors"]


@pytest.mark.parametrize("revision,filename", [("main", "config.json"), ("a" * 40, "../config.json")])
def test_mutable_revisions_and_traversing_paths_are_rejected(tmp_path, revision, filename):
    value = {"schema_version": 1, "models": {"m": {"repository": "test/m", "revision": revision,
             "files": {filename: {"size": 1, "sha256": "a" * 64}}}}}
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_lock(path)
