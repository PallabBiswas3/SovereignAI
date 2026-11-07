"""Provisioning must distinguish absent tags from daemon/read errors."""
import hashlib
import importlib.util
from pathlib import Path
import runpy
import subprocess

import pytest


@pytest.mark.parametrize("status,content,allowed", [
    (44, b"", True),
    (0, b"locked manifest", True),
    (0, b"conflicting manifest", False),
    (1, b"", False),
    (125, b"", False),
])
def test_existing_tag_inspection_fails_closed(monkeypatch, status, content, allowed):
    module = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/provision_ollama_models.py"))
    lock = {"models": {"test:tag": {"manifest_path": "manifests/test/tag",
            "manifest_sha256": hashlib.sha256(b"locked manifest").hexdigest()}}}
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, status, stdout=content))
    if allowed:
        module["check_existing_tags"]("test-container", lock)
    else:
        with pytest.raises(RuntimeError, match="no overwrite"):
            module["check_existing_tags"]("test-container", lock)


@pytest.mark.parametrize("storage_only", [False, True])
def test_volume_verification_uses_nested_models_directory(monkeypatch, storage_only):
    path = Path(__file__).resolve().parents[2] / "scripts/provision_ollama_models.py"
    spec = importlib.util.spec_from_file_location("ollama_provision", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module, "docker", lambda *args, **kwargs: calls.append(args))
    module.verify_volume("test-container", "preserved-volume", Path("lock.json"), storage_only=storage_only)
    command = calls[0]
    assert command[command.index("--root") + 1] == "/models/models"
    assert "type=volume,source=preserved-volume,target=/models,readonly" in command
    assert ("--api" in command) is not storage_only
