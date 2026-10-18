"""Native long-path I/O must retain resource confinement and relative API paths."""
import asyncio
import io
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import UploadFile

from app.api import files
from app.core.config import get_settings
from app.tools.file_tools import ReadFileTool, SafeWorkspace, principal_workspace
from test_file_isolation import actor


@pytest.mark.skipif(os.name != 'nt', reason='Windows native long-path regression')
def test_long_upload_can_be_listed_and_read_only_by_its_owner(tmp_path, monkeypatch):
    root = tmp_path / ('nested-' + 'a' * 80)
    settings = SimpleNamespace(workspace_root=root, max_upload_mb=1, auth_mode='local')
    monkeypatch.setattr(files, 'get_settings', lambda: settings)
    monkeypatch.setenv('SOVEREIGN_AUTH_MODE', 'local')
    get_settings.cache_clear()
    try:
        owner = actor()
        stranger = actor(org='foreign', user='foreign')
        uploaded = asyncio.run(files.upload_file(UploadFile(filename='synthetic-private.txt',
            file=io.BytesIO(b'Synthetic private evidence')), owner))
        workspace = principal_workspace(root, owner)
        native = workspace.resolve(uploaded['path'], must_exist=True)
        assert len(str(native)) > 260
        assert native.read_text() == 'Synthetic private evidence'
        assert not uploaded['path'].startswith('\\\\?\\')
        assert [item['path'] for item in asyncio.run(files.list_files(owner))] == [uploaded['path']]
        assert asyncio.run(files.list_files(stranger)) == []
        assert asyncio.run(ReadFileTool(workspace).execute({'path':uploaded['path']})).success
        foreign = principal_workspace(root, stranger)
        for path in [str(native), str(native)[4:], '../../outside.txt']:
            with pytest.raises(ValueError, match='traversal'):
                foreign.resolve(path, must_exist=True)
    finally:
        get_settings.cache_clear()


@pytest.mark.skipif(os.name != 'nt', reason='Windows volume syntax regression')
def test_docker_mount_uses_host_path_and_retains_container_restrictions(tmp_path, monkeypatch):
    from app.sandbox.executor import DockerSandboxExecutor
    import app.sandbox.executor as sandbox
    workspace = SafeWorkspace(tmp_path/'sandbox')
    monkeypatch.setattr(sandbox.shutil, 'which', lambda _: 'docker')
    commands=[]
    class Process:
        returncode=0
        async def communicate(self): return b'',b''
    async def create(*args, **kwargs):
        commands.append(args)
        return Process()
    monkeypatch.setattr(sandbox.asyncio,'create_subprocess_exec',create)
    result=asyncio.run(DockerSandboxExecutor(workspace.root).execute('print("synthetic")'))
    assert result.executed
    command=commands[0]
    assert not command[command.index('-v')+1].startswith('\\\\?\\')
    for flag in ['--read-only','--cap-drop','--security-opt','--network','--user']:
        assert flag in command
