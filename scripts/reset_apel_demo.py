from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings
from app.core.database import SessionLocal, init_db
from app.demo.apel import ApelDemoService


def main() -> None:
    settings = get_settings()
    if not settings.demo_org_enabled:
        raise SystemExit("APEL demo reset is disabled. Set SOVEREIGN_DEMO_ORG_ENABLED=true explicitly.")
    init_db()
    with SessionLocal() as session:
        result = ApelDemoService(session, ROOT / "demo" / "apel", ROOT / "demo" / "apel" / "generated").reset()
    print(f"Reset APEL: {result['users']} users, {result['runs']} runs, {result['documents']} documents removed.")


if __name__ == "__main__":
    main()
