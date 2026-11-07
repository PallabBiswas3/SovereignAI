"""Recovery safeguards must fail before changing an existing deployment."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_docker(monkeypatch, module, *, running="", existing=False):
    commands = []
    config = {"name": "recovery", "services": {"gateway": {"image": "local-helper:test"}},
              "volumes": {"workspace": {"name": "recovery_workspace"}}}

    def run(argv, **kwargs):
        commands.append(argv)
        if "config" in argv:
            return SimpleNamespace(stdout=json.dumps(config), returncode=0)
        if "ps" in argv:
            return SimpleNamespace(stdout=running, returncode=0)
        if argv[1:3] == ["volume", "inspect"]:
            return SimpleNamespace(stdout="", returncode=0 if existing else 1)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    return commands


def backup_fixture(tmp_path):
    archive = tmp_path / "workspace.tar.gz"
    archive.write_bytes(b"synthetic-archive-not-extracted")
    record = {"volumes": {"workspace": {"archive": archive.name,
              "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}}}
    (tmp_path / "manifest.json").write_text(json.dumps(record))


def test_backup_refuses_active_services_before_copying(monkeypatch, tmp_path):
    module = load_script("docker_backup")
    calls = fake_docker(monkeypatch, module, running="active-container")
    monkeypatch.setattr(sys, "argv", ["docker_backup", "backup", str(tmp_path / "new")])
    with pytest.raises(SystemExit, match="Stop the whole project"):
        module.main()
    assert not (tmp_path / "new").exists()
    assert not any(command[1] == "run" for command in calls)


def test_restore_refuses_existing_target_without_creating_or_extracting(monkeypatch, tmp_path):
    module = load_script("docker_backup")
    backup_fixture(tmp_path)
    calls = fake_docker(monkeypatch, module, existing=True)
    monkeypatch.setattr(sys, "argv", ["docker_backup", "restore", str(tmp_path), "--project-name", "recovery"])
    with pytest.raises(SystemExit, match="already exists"):
        module.main()
    assert not any(command[1] == "run" or command[1:3] == ["volume", "create"] for command in calls)


def test_corrupt_restore_is_rejected_before_any_volume_creation(monkeypatch, tmp_path):
    module = load_script("docker_backup")
    backup_fixture(tmp_path)
    (tmp_path / "workspace.tar.gz").write_bytes(b"damaged")
    calls = fake_docker(monkeypatch, module)
    monkeypatch.setattr(sys, "argv", ["docker_backup", "restore", str(tmp_path), "--project-name", "recovery"])
    with pytest.raises(SystemExit, match="verification failed"):
        module.main()
    assert not any(command[1] == "run" or command[1:3] == ["volume", "create"] for command in calls)


def test_smoke_reports_unavailable_engine_as_failure(monkeypatch, capsys):
    module = load_script("docker_smoke")

    def run(argv, **kwargs):
        if argv == ["docker", "info"]:
            raise subprocess.CalledProcessError(1, argv)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["docker_smoke"])
    assert module.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert "docker_engine" not in report["checks"]


def test_container_probe_can_be_launched_by_absolute_path(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / "backend/scripts/docker_probe.py"), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--inference" in result.stdout


def test_graphrag_secret_wrapper_starts_exported_application():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to verify the GraphRAG runtime wrapper")
    source = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const env = { PORT: '3100', HOST: '0.0.0.0' };
let listening = false;
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
  process: {env}, console: {log() {}},
  require(name) {
    if (name === 'node:fs') return {readFileSync() {return 'synthetic-test-token-'.repeat(3)}};
    assert.equal(name, './dist/server.js');
    assert.ok(env.GRAPHRAG_SERVICE_TOKEN.length >= 32);
    assert.ok(env.SUPABASE_SECRET_KEY.length >= 32);
    return {app: {listen(port, host, callback) {
      assert.equal(port, 3100); assert.equal(host, '0.0.0.0');
      listening = true; callback();
    }}};
  }
});
assert.ok(listening, 'wrapper must start the exported HTTP application');
"""
    result = subprocess.run([node, "-e", source, str(ROOT / "deploy/graphrag-entrypoint.cjs")],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
