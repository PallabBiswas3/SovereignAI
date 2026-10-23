"""Explicit operator CLI for a scoped local account; no default passwords."""
from __future__ import annotations

import argparse
from getpass import getpass
import json
from uuid import uuid4

from sqlalchemy import select

from app.core.database import SessionLocal, UserRecord, init_db
from app.identity.models import Role
from app.identity.passwords import PasswordHasher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True)
    parser.add_argument("--organization", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--department", required=True)
    parser.add_argument("--role", action="append", choices=[r.value for r in Role], required=True)
    args = parser.parse_args()
    if any("*" in value or not value.strip() for value in (args.email, args.organization, args.workspace, args.department)):
        raise SystemExit("Explicit non-wildcard identity and scope required")
    password = getpass("New password (at least 12 characters): ")
    if len(password) < 12 or password != getpass("Confirm password: "):
        raise SystemExit("Password too short or confirmation differs")
    password_hash = PasswordHasher().hash(password)
    init_db()
    with SessionLocal() as db:
        if db.scalar(select(UserRecord.id).where(UserRecord.email_normalized == args.email.strip().lower())):
            raise SystemExit("Account already exists; this command never overwrites accounts")
        db.add(UserRecord(id=str(uuid4()), email=args.email.strip(), email_normalized=args.email.strip().lower(),
                          display_name=args.email.strip(), organization_id=args.organization,
                          workspace_ids_json=json.dumps([args.workspace]), department_ids_json=json.dumps([args.department]),
                          roles_json=json.dumps(args.role), clearance="INTERNAL", password_hash=password_hash))
        db.commit()
    print("Scoped account created. Password and session credentials were not logged.")


if __name__ == "__main__":
    main()
