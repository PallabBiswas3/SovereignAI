"""Regressions for backend defects fixed on 8 November 2026.

Account enumeration through login timing, anonymous model-status probes, sandbox
containers surviving their timeout, and host files reachable through symlinks
written by sandboxed code.
"""
import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import get_settings


# Login timing ------------------------------------------------------------------

class _Query:
    def __init__(self, user):
        self.user = user

    def filter(self, *_):
        return self

    def first(self):
        return self.user


class _Session:
    def __init__(self, user):
        self.user = user

    def query(self, _):
        return _Query(self.user)


class _SpyHasher:
    iterations = 100_000

    def __init__(self):
        self.verified = []

    def hash(self, password):
        return f"dummy:{len(password)}"

    def verify(self, password, encoded):
        self.verified.append(encoded)
        return encoded == "stored-hash" and password == "correct-password"


def _provider(user):
    from app.identity.provider import LocalIdentityProvider
    provider = LocalIdentityProvider.__new__(LocalIdentityProvider)
    provider.session, provider.password_hasher, provider.role_permissions = _Session(user), _SpyHasher(), {}
    return provider


class _User:
    def __init__(self, enabled=True):
        self.enabled, self.password_hash = enabled, "stored-hash"


@pytest.mark.parametrize("user", [None, _User(enabled=False)])
def test_unknown_or_disabled_account_still_performs_key_derivation(user):
    provider = _provider(user)
    assert provider.authenticate("nobody@example.internal", "wrong-password") is None
    assert len(provider.password_hasher.verified) == 1  # same work as a real check


def test_real_account_login_behaviour_is_unchanged():
    user = _User()
    assert _provider(user).authenticate("real@example.internal", "correct-password") is user
    assert _provider(user).authenticate("real@example.internal", "wrong-password") is None


def test_dummy_hash_is_a_valid_pbkdf2_hash():
    from app.identity.passwords import PasswordHasher
    from app.identity.provider import _dummy_hash
    hasher = PasswordHasher(iterations=100_000)
    encoded = _dummy_hash(hasher)
    assert encoded.startswith("pbkdf2-sha256$100000$")
    assert hasher.verify("anything-at-all", encoded) is False


# Model status authentication ---------------------------------------------------

def test_model_endpoints_require_authentication_in_local_mode(monkeypatch):
    monkeypatch.setenv("SOVEREIGN_AUTH_MODE", "local")
    get_settings.cache_clear()
    try:
        from app.api.models import router
        app = FastAPI(); app.include_router(router)
        client = TestClient(app)
        for path in ("/api/models", "/api/models/status", "/api/models/route?request=hello"):
            assert client.get(path).status_code == 401, path
    finally:
        get_settings.cache_clear()


# Sandbox timeout and symlink outputs -------------------------------------------

def test_timed_out_sandbox_container_is_force_removed(tmp_path, monkeypatch):
    import app.sandbox.executor as sandbox
    commands = []

    class Hanging:
        returncode = None

        async def communicate(self):
            if self.returncode is None:
                await asyncio.sleep(3600)
            return b"", b""

        def kill(self):
            self.returncode = -9

    class Remover:
        returncode = 0

        async def wait(self):
            return 0

    async def create(*args, **kwargs):
        commands.append(args)
        return Hanging() if args[1] == "run" else Remover()

    monkeypatch.setattr(sandbox.shutil, "which", lambda _: "docker")
    monkeypatch.setattr(sandbox.asyncio, "create_subprocess_exec", create)
    result = asyncio.run(sandbox.DockerSandboxExecutor(tmp_path, timeout=0.05).execute("while True: pass"))
    run, remove = commands
    name = run[run.index("--name") + 1]
    assert name.startswith("sovereign-sandbox-")
    assert remove == ("docker", "rm", "--force", name)
    assert "timeout" in result.stderr and "could not be confirmed" not in result.stderr
    # Existing isolation flags are unchanged.
    for flag in ("--network", "none", "--read-only", "--cap-drop", "ALL", "no-new-privileges"):
        assert flag in run


def _pretend_symlink(monkeypatch, name):
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: self.name == name or original(self))


def test_sandbox_output_listing_excludes_symlinks(tmp_path, monkeypatch):
    import app.sandbox.executor as sandbox

    class Process:
        returncode = 0

        async def communicate(self):
            run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
            (run_dir / "result.csv").write_text("a,b\n1,2\n")
            (run_dir / "leak.txt").write_text("stands in for a link to a host file")
            return b"", b""

    async def create(*args, **kwargs):
        return Process()

    monkeypatch.setattr(sandbox.shutil, "which", lambda _: "docker")
    monkeypatch.setattr(sandbox.asyncio, "create_subprocess_exec", create)
    _pretend_symlink(monkeypatch, "leak.txt")
    result = asyncio.run(sandbox.DockerSandboxExecutor(tmp_path).execute("pass"))
    assert result.generated_files == ["result.csv"]


def test_coding_workflow_never_copies_a_symlinked_output(tmp_path, monkeypatch):
    from app.sandbox.executor import SandboxResult
    from app.workflows.coding import CodingWorkflow
    source = tmp_path / "run"; source.mkdir()
    (source / "result.csv").write_text("a,b\n1,2\n")
    (source / "secret.txt").write_text("stands in for a link to a host file")
    destination = tmp_path / "out"; destination.mkdir()
    _pretend_symlink(monkeypatch, "secret.txt")
    execution = SandboxResult(exit_code=0, executed=True, output_directory=str(source),
                              generated_files=["result.csv", "secret.txt"])
    collected = CodingWorkflow._collect_outputs(execution, tmp_path / "input.csv", destination)
    assert [Path(item).name for item in collected] == ["result.csv"]
    assert not (destination / "secret.txt").exists()


def test_real_symlink_is_not_followed_when_supported(tmp_path):
    from app.sandbox.executor import SandboxResult
    from app.workflows.coding import CodingWorkflow
    host_secret = tmp_path / "host-secret.txt"; host_secret.write_text("HOST SECRET")
    source = tmp_path / "run"; source.mkdir()
    try:
        (source / "out.txt").symlink_to(host_secret)
    except OSError:
        pytest.skip("symlink creation is not permitted on this host")
    destination = tmp_path / "out"; destination.mkdir()
    execution = SandboxResult(exit_code=0, executed=True, output_directory=str(source), generated_files=["out.txt"])
    assert CodingWorkflow._collect_outputs(execution, tmp_path / "input.csv", destination) == []
