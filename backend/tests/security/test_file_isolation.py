"""Raw file readers must authorize paths before parsing, caching or generation."""
import asyncio
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile

from app.api import demo, files, knowledge, ocr, tasks, vision
from app.core.config import get_settings
from app.identity.models import ClearanceLevel, Permission, Principal, Role
from app.tools.file_tools import principal_workspace
from app.tools.registry import create_default_registry


def actor(org="org-a", workspace="workspace-a", user="user-a"):
    return Principal(user_id=user, email="synthetic@example.invalid", display_name="Synthetic",
                     organization_id=org, workspace_ids=[workspace], department_ids=["engineering"],
                     roles=[Role.engineer], permissions=list(Permission), clearance=ClearanceLevel.restricted)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SOVEREIGN_AUTH_MODE", "local")
    monkeypatch.setenv("SOVEREIGN_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("SOVEREIGN_KNOWLEDGE_ROOT", str(tmp_path / "knowledge"))
    monkeypatch.setenv("SOVEREIGN_CACHE_ENABLED", "false")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


def upload(principal, name="report.md", content=b"synthetic-private-data"):
    return asyncio.run(files.upload_file(UploadFile(filename=name, file=io.BytesIO(content)), principal))


@pytest.mark.parametrize("other", [actor(org="org-b"), actor(workspace="workspace-b"), actor(user="user-b")])
def test_upload_namespaces_and_overwrites_are_isolated(settings, other):
    owner = actor()
    first = upload(owner)
    second = upload(other, content=b"other contents")
    first_path = principal_workspace(settings.workspace_root, owner).resolve(first["path"], must_exist=True)
    assert first_path.read_bytes() == b"synthetic-private-data"
    listed = asyncio.run(files.list_files(other))
    assert [item["path"] for item in listed] == [second["path"]]
    with pytest.raises((FileNotFoundError, ValueError)):
        principal_workspace(settings.workspace_root, other).resolve(first["path"], must_exist=True)
    duplicate = upload(owner, content=b"new own version")
    assert duplicate["path"] != first["path"]
    assert first_path.read_bytes() == b"synthetic-private-data"


@pytest.mark.parametrize("consumer,extension", [
    ("ocr", ".png"), ("vision", ".png"), ("knowledge", ".md"),
    ("inspection_demo", ".md"), ("coding_demo", ".csv"),
    ("task", ".json"), ("start", ".json"),
])
@pytest.mark.parametrize("absolute", [False, True])
def test_every_http_file_consumer_rejects_foreign_file_before_use(settings, consumer, extension, absolute, monkeypatch):
    owner, stranger = actor(), actor(org="org-b", user="user-b")
    item = upload(owner, "report" + extension)
    path = str(principal_workspace(settings.workspace_root, owner).resolve(item["path"])) if absolute else item["path"]
    def must_not_run(*args, **kwargs):
        pytest.fail("Unauthorized input reached parsing/model/cache initialization")
    monkeypatch.setattr(ocr, "LocalOCRService", must_not_run)
    monkeypatch.setattr(vision, "ModelRegistry", must_not_run)
    monkeypatch.setattr(knowledge, "configured_embedding_provider", must_not_run)
    monkeypatch.setattr(demo, "configured_embedding_provider", must_not_run)
    monkeypatch.setattr(tasks, "ModelRouter", must_not_run)
    with pytest.raises(HTTPException) as denied:
        if consumer == "ocr":
            asyncio.run(ocr.run_ocr(ocr.OCRRequest(path=path), stranger))
        elif consumer == "vision":
            asyncio.run(vision.analyze_image(vision.VisionRequest(path=path), stranger))
        elif consumer == "knowledge":
            asyncio.run(knowledge.ingest(knowledge.IngestRequest(path=path), stranger, None))
        elif consumer == "inspection_demo":
            asyncio.run(demo.run_inspection_demo(demo.InspectionDemoRequest(inspection_path=path), stranger, None))
        elif consumer == "coding_demo":
            asyncio.run(demo.run_coding_demo(demo.CodingDemoRequest(csv_path=path), stranger, None))
        elif consumer == "task":
            asyncio.run(tasks.create_task(tasks.CreateTaskRequest(request="Diagnose process", attachments=[path]), stranger, None))
        else:
            asyncio.run(tasks.start_task(tasks.CreateTaskRequest(request="Diagnose process", attachments=[path]), stranger, None))
    assert denied.value.status_code in {400, 404}


def test_own_upload_can_be_consumed_and_routed(settings, monkeypatch):
    owner = actor()
    item = upload(owner, "report.png")
    class SyntheticOCR:
        def __init__(self, **kwargs): pass
        def extract(self, path):
            assert path.read_bytes() == b"synthetic-private-data"
            return SimpleNamespace(model_dump=lambda: {"text": "synthetic OCR", "available": True})
    monkeypatch.setattr(ocr, "LocalOCRService", SyntheticOCR)
    assert asyncio.run(ocr.run_ocr(ocr.OCRRequest(path=item["path"]), owner))["available"]
    tasks._validate_attachments(tasks.CreateTaskRequest(request="Inspect report", attachments=[item["path"]]), settings, owner)


def test_tools_and_approved_action_registry_cannot_read_foreign_or_global_artifacts(settings):
    owner, stranger = actor(), actor(user="user-b")
    item = upload(owner)
    global_artifact = settings.workspace_root / "artifacts/private.md"
    global_artifact.parent.mkdir(parents=True)
    global_artifact.write_text("synthetic artifact")
    registry = create_default_registry(settings.workspace_root, stranger)
    for path in (item["path"], str(principal_workspace(settings.workspace_root, owner).resolve(item["path"])), str(global_artifact), "../report.md"):
        assert not asyncio.run(registry.get("read_file").execute({"path": path})).success
        assert not asyncio.run(registry.get("run_python").execute({"code": "pass", "input_files": [path]})).success
    assert not asyncio.run(registry.get("search_files").execute({"query": "synthetic-private-data"})).output


def test_legacy_global_uploads_and_missing_scope_are_not_assumed_authorized(settings):
    from app.identity.models import Principal
    path = settings.workspace_root / "uploads/old.md"
    path.parent.mkdir(parents=True)
    path.write_text("legacy")
    assert asyncio.run(files.list_files(actor())) == []
    with pytest.raises(ValueError):
        principal_workspace(settings.workspace_root, None)
    with pytest.raises(ValueError):
        principal_workspace(settings.workspace_root, actor(workspace="*"))


def test_inspection_cache_is_bound_to_effective_authorization(settings, tmp_path):
    from app.workflows.inspection import InspectionWorkflow
    class ReachedExtractor(Exception): pass
    class Extractor:
        def extract(self, path): raise ReachedExtractor()
    class Cache:
        keys = []
        def get(self, namespace, key):
            self.keys.append(key)
            return None
    path = tmp_path / "common-template.md"
    path.write_text("identical authorized input bytes")
    cache = Cache()
    for principal in (actor(), actor(org="org-b"), actor(workspace="workspace-b")):
        retriever = SimpleNamespace(principal=principal, collection_version=lambda: "same-collection")
        workflow = InspectionWorkflow(retriever, Extractor(), cache)
        with pytest.raises(ReachedExtractor):
            workflow.analyze(path, tmp_path / "unused.docx")
    assert len(set(cache.keys)) == 3
