from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.database import Base, KnowledgeDocument, UserRecord
from app.demo.apel import ApelDemoService
from app.identity.provider import LocalIdentityProvider
from app.rag.embeddings import LocalHashEmbeddingProvider
from app.rag.retrieval import LocalRetriever


ROOT = Path(__file__).resolve().parents[1]


def _digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def test_apel_generator_is_coherent_and_deterministic(tmp_path: Path) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    generated = tmp_path / "apel" / "generated"
    with Session(engine) as session:
        service = ApelDemoService(session, ROOT / "demo" / "apel", generated)
        first = service.generate()
        first_digest = _digest(generated)
        second = service.generate()
        assert _digest(generated) == first_digest
        assert first == second
        assert len(service.assets) == 20
        assert 30 <= len(first) <= 60
        pump = (generated / "engineering/datasheets/Pump-102_Datasheet.md").read_text()
        inspection = (generated / "maintenance/inspections/Pump-102_Inspection_2026-09-01.md").read_text()
        assert "4.8-5.5" in pump and "9.0 mm/s" in pump
        assert "5.1 bar" in inspection and "7.4 mm/s" in inspection


def test_apel_seed_acl_and_reset(tmp_path: Path) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    generated = tmp_path / "apel" / "generated"
    with Session(engine) as session:
        service = ApelDemoService(session, ROOT / "demo" / "apel", generated)
        result = service.seed()
        assert result == {"assets": 20, "files": 55, "users": 7}
        assert session.query(KnowledgeDocument).count() == 55
        assert session.query(UserRecord).count() == 7

        provider = LocalIdentityProvider(session, ROOT / "config" / "access.yaml")
        engineer = provider.principal_for_user("apel-maint-001")
        assert engineer is not None
        results = LocalRetriever(session, LocalHashEmbeddingProvider(), principal=engineer).search("FIN-XC-926 executive compensation", 20)
        assert all("FIN-XC-926" not in item.text for item in results)
        assert all(item.source.get("department") != "finance" for item in results)

        removed = service.reset()
        assert removed["users"] == 7 and removed["documents"] == 55
        assert session.query(UserRecord).count() == 0
        assert session.query(KnowledgeDocument).count() == 0


def test_apel_evaluation_dataset_has_required_coverage() -> None:
    payload = json.loads((ROOT / "demo/apel/evaluation/v1.json").read_text(encoding="utf-8"))
    questions = payload["questions"]
    assert len(questions) == 50
    kinds = {item["kind"] for item in questions}
    assert {"exact", "numeric", "cross_document", "missing_evidence", "unauthorized", "unit_conflict"} <= kinds
