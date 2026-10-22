import asyncio
import hashlib
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.artifacts.service import ArtifactService
from app.core.database import Base, KnowledgeDocument, ArtifactRecord
from app.identity.models import DocumentACL, ClearanceLevel
from app.rag.embeddings import LocalHashEmbeddingProvider
from app.rag.ingestion import KnowledgeIngestionService
from app.tools.application_tools import GenerateDocxTool, GenerateXlsxTool, GeneratePptxTool


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.mark.parametrize("tool,args", [(GenerateDocxTool, {"sections": []}),
    (GenerateXlsxTool, {"rows": []}), (GeneratePptxTool, {"slides": []})])
def test_same_download_name_cannot_overwrite_another_artifact(db, tmp_path, tool, args):
    service = ArtifactService(db, tmp_path)
    first = asyncio.run(tool(service).execute({"title": "First", "filename": "shared", **args}))
    record = db.get(ArtifactRecord, first.output["artifact"]["id"])
    original = service.resolve(record).read_bytes()
    second = asyncio.run(tool(service).execute({"title": "Second", "filename": "shared", **args}))
    other = db.get(ArtifactRecord, second.output["artifact"]["id"])
    assert record.path != other.path
    assert record.name == other.name
    assert service.resolve(record).read_bytes() == original


@pytest.mark.parametrize("updates", [{"owner_id": "user-b"}, {"department_id": "finance"},
    {"classification": ClearanceLevel.public}, {"organization_id": "org-b"}, {"workspace_id": "workspace-b"}])
def test_identical_bytes_do_not_reassign_existing_document_acl(db, tmp_path, updates):
    path = tmp_path / "synthetic.txt"
    path.write_text("Synthetic evidence about a pump inspection.")
    acl = DocumentACL(organization_id="org-a", workspace_id="workspace-a", owner_id="user-a",
                      department_id="engineering", classification=ClearanceLevel.confidential)
    service = KnowledgeIngestionService(db, LocalHashEmbeddingProvider())
    document = service.ingest(path, acl=acl, require_acl=True)
    before = document.metadata_json
    with pytest.raises(ValueError, match="DOCUMENT_ALREADY_SCOPED"):
        service.ingest(path, acl=acl.model_copy(update=updates), require_acl=True)
    assert document.metadata_json == before
    assert service.ingest(path, acl=acl, require_acl=True).id == document.id


def test_identical_upload_does_not_adopt_unscoped_legacy_document(db, tmp_path):
    path = tmp_path / "legacy.txt"
    path.write_text("Synthetic legacy evidence.")
    service = KnowledgeIngestionService(db, LocalHashEmbeddingProvider())
    document = service.ingest(path)
    with pytest.raises(ValueError, match="DOCUMENT_ALREADY_SCOPED"):
        service.ingest(path, acl=DocumentACL(organization_id="org-a", workspace_id="workspace-a", owner_id="user-a"), require_acl=True)
    assert document.organization_id is None
