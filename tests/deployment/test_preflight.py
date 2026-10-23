import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_secrets_are_consistent_preserved_and_tampering_is_detected(tmp_path):
    prepare = script("prepare_production")
    preflight = script("preflight_production")
    assert len(prepare.initialize(tmp_path)) == 6
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert prepare.initialize(tmp_path) == []
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert all(preflight.secret_checks(tmp_path).values())
    path = tmp_path / "graph_jwt_secret"
    path.chmod(0o600)
    path.write_text("different-signing-key-" * 3)
    assert preflight.secret_checks(tmp_path)["graph_jwt_valid"] is False


def test_partial_credentials_never_trigger_automatic_rotation(tmp_path):
    (tmp_path / "postgres_password").write_text("existing-value")
    with pytest.raises(ValueError, match="Partial secret set"):
        script("prepare_production").initialize(tmp_path)
    assert (tmp_path / "postgres_password").read_text() == "existing-value"


def test_incomplete_model_directory_fails_preflight(tmp_path):
    assert script("preflight_production").model_files_present(tmp_path) is False
    (tmp_path / "config.json").write_text("{}")
    assert script("preflight_production").model_files_present(tmp_path) is False
