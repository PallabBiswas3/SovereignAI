import json
import sys

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.core.database import UserRecord
from app.identity import provision
from app.identity.passwords import PasswordHasher


@pytest.fixture
def accounts(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'accounts.db').as_posix()}")
    UserRecord.__table__.create(engine)
    factory = sessionmaker(engine)
    monkeypatch.setattr(provision, "init_db", lambda: None)
    monkeypatch.setattr(provision, "SessionLocal", factory)
    monkeypatch.setattr(provision, "getpass", lambda _: "unique-test-password-2026")
    monkeypatch.setattr(sys, "argv", ["provision", "--email", "operator@example.internal",
                        "--organization", "org", "--workspace", "engineering", "--department", "operations", "--role", "ENGINEER"])
    yield factory
    engine.dispose()


def test_provisioning_creates_only_requested_scope_and_hashes_password(accounts, capsys):
    provision.main()
    with accounts() as db:
        user = db.scalar(select(UserRecord))
        assert user.organization_id == "org"
        assert json.loads(user.workspace_ids_json) == ["engineering"]
        assert json.loads(user.department_ids_json) == ["operations"]
        assert json.loads(user.roles_json) == ["ENGINEER"]
        assert user.clearance == "INTERNAL"
        assert PasswordHasher().verify("unique-test-password-2026", user.password_hash)
    assert "unique-test-password" not in capsys.readouterr().out


def test_duplicate_account_never_changes_existing_credentials(accounts):
    provision.main()
    with accounts() as db:
        original = db.scalar(select(UserRecord)).password_hash
    with pytest.raises(SystemExit, match="already exists"):
        provision.main()
    with accounts() as db:
        assert db.scalar(select(UserRecord)).password_hash == original


def test_wildcard_scope_is_rejected_before_account_creation(accounts, monkeypatch):
    argv = list(sys.argv)
    argv[argv.index("--workspace") + 1] = "*"
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="non-wildcard"):
        provision.main()
    with accounts() as db:
        assert db.scalar(select(UserRecord)) is None
