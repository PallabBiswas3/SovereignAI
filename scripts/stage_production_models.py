"""Stage explicit local model snapshots; network use requires --download.

The resolved commit for each model is recorded alongside the staged files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.monitoring.model_manifest import load_lock, verify_directory


def main():
    from huggingface_hub import snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Allow downloading missing official model snapshots during provisioning")
    args = parser.parse_args()
    destination = ROOT / "deploy/models"
    destination.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, model in load_lock(ROOT / "deploy/model-lock.json").items():
        target = destination / name
        if target.exists():
            result = verify_directory(target, model)
            if not result["valid"]:
                failed.append(name)
            print(json.dumps({"model": name, "existing_preserved": True, **result}))
            continue
        try:
            snapshot = Path(snapshot_download(model["repository"], revision=model["revision"],
                local_files_only=not args.download, allow_patterns=list(model["files"])))
            with tempfile.TemporaryDirectory(prefix="staging-", dir=destination) as temporary:
                staging = Path(temporary) / name
                staging.mkdir()
                for filename in model["files"]:
                    path = staging / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(snapshot / filename, path)
                (staging / "sovereign-snapshot.json").write_text(json.dumps({
                    "repository": model["repository"], "revision": model["revision"]}, indent=2), encoding="utf-8")
                result = verify_directory(staging, model)
                if not result["valid"]:
                    raise ValueError("Downloaded snapshot failed the locked file checks")
                staging.rename(target)
            print(json.dumps({"model": name, **result}))
        except Exception as exc:
            failed.append(name)
            print(f"{name}: staging failed ({type(exc).__name__}); existing directories preserved")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
