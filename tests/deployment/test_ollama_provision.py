"""Provisioning must distinguish absent tags from daemon/read errors."""
import hashlib
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
