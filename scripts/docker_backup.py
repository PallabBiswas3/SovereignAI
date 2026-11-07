"""Cold backup/restore of Compose volumes, without overwriting existing volumes.

Stop the entire project before backup. Restore only to a NEW project name and
restore matching configuration/secrets separately. Archives include private data
and CA keys: protect the output directory as sensitive material.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["backup", "restore"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--env-file", default="deploy/.env")
    parser.add_argument("--project-name", help="Required NEW project name for restore")
    args = parser.parse_args()
    compose = ["docker", "compose", "--env-file", args.env_file, "-f", "compose.yaml"]
    if args.project_name:
        compose += ["--project-name", args.project_name]

    def run(argv, **kwargs):
        return subprocess.run(argv, cwd=ROOT, check=True, timeout=kwargs.pop("timeout", 1200), **kwargs)

    config = json.loads(run(compose + ["config", "--format", "json"], capture_output=True, text=True).stdout)
    running = run(compose + ["ps", "--status", "running", "-q"], capture_output=True, text=True).stdout.strip()
    if running:
        raise SystemExit("Stop the whole project before backup/restore; no data was copied")
    helper = config["services"]["gateway"]["image"]
    # Require the helper locally: backup/restore must not silently pull images.
    run(["docker", "image", "inspect", helper], stdout=subprocess.DEVNULL)
    directory = args.directory.resolve()
    volumes = {key: value["name"] for key, value in config["volumes"].items()}
    common = ["docker", "run", "--rm", "--pull", "never", "--network", "none", "--read-only",
              "--user", "0:0", "--cap-drop", "ALL", "--cap-add", "DAC_OVERRIDE",
              "--security-opt", "no-new-privileges:true", "--entrypoint", "tar"]
    if args.action == "backup":
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {"project": config["name"], "volumes": {}}
        for key, volume in volumes.items():
            run(["docker", "volume", "inspect", volume], stdout=subprocess.DEVNULL)
            archive = directory / f"{key}.tar.gz"
            with archive.open("xb") as stream:
                run(common + ["--mount", f"type=volume,source={volume},target=/volume,readonly",
                              helper, "-czf", "-", "-C", "/volume", "."], stdout=stream)
            manifest["volumes"][key] = {"archive": archive.name, "sha256": sha256(archive)}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        if not args.project_name:
            raise SystemExit("Restore requires --project-name for a NEW isolated project")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if set(manifest["volumes"]) != set(volumes):
            raise SystemExit("Volume inventory mismatch; use the matching release")
        for key, volume in volumes.items():
            record = manifest["volumes"][key]
            # Only the fixed names emitted by backup are accepted.
            if record["archive"] != f"{key}.tar.gz" or sha256(directory / record["archive"]) != record["sha256"]:
                raise SystemExit("Backup verification failed; no volume was restored")
            found = subprocess.run(["docker", "volume", "inspect", volume], capture_output=True)
            if found.returncode == 0:
                raise SystemExit("Target volume already exists; restore refuses to overwrite it")
        for key, volume in volumes.items():
            run(["docker", "volume", "create", "--label", f"com.docker.compose.project={config['name']}",
                 "--label", f"com.docker.compose.volume={key}", volume], stdout=subprocess.DEVNULL)
            with (directory / manifest["volumes"][key]["archive"]).open("rb") as stream:
                run(common + ["--cap-add", "CHOWN", "--cap-add", "FOWNER", "-i", "--mount",
                              f"type=volume,source={volume},target=/volume", helper,
                              "-xzf", "-", "-C", "/volume"], stdin=stream)
    print(f"{args.action} completed for {len(volumes)} volumes; configuration, secrets and bind-mounted models are separate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
