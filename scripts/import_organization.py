from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings
from app.core.database import SessionLocal, init_db
from app.organizations.importer import OrganizationImportError, OrganizationPackImporter
from app.organizations.loader import OrganizationPackError, OrganizationPackLoader


def resolve_pack_path(value: str, organizations_root: Path) -> Path:
    root = organizations_root.resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        direct = (ROOT / candidate).resolve()
        candidate = direct if direct == root or root in direct.parents else (root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError(f"Pack must be a child directory of {root}")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and import a versioned SovereignAI Organization Pack.",
    )
    parser.add_argument("--pack", required=True, help="Pack name or path under organizations/")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate files, references, Workcells, and database collisions without importing.",
    )
    parser.add_argument(
        "--rotate-local-passwords", action="store_true",
        help="Re-hash existing local-user passwords from their declared environment variables.",
    )
    args = parser.parse_args()
    settings = get_settings()
    try:
        pack_path = resolve_pack_path(args.pack, settings.organizations_root)
        pack = OrganizationPackLoader(
            pack_path, max_file_bytes=settings.max_upload_mb * 1024 * 1024,
        ).load()
        init_db()
        with SessionLocal() as session:
            report = OrganizationPackImporter(session, settings=settings).import_pack(
                pack, dry_run=args.dry_run,
                rotate_local_passwords=args.rotate_local_passwords,
            )
    except (ValueError, OrganizationPackError, OrganizationImportError) as exc:
        raise SystemExit(f"Organization Pack rejected: {exc}") from exc
    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
