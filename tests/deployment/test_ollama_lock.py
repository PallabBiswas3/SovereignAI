import hashlib
import json
from pathlib import Path
import runpy


def test_ollama_manifest_and_blob_tampering_are_detected(tmp_path):
    verify = runpy.run_path(str(Path(__file__).with_name("ollama_volume_verify.py")))["verify"]
    content = b"synthetic model bytes"
    manifest = b'{"synthetic":true}'
    digest = hashlib.sha256(content).hexdigest()
    (tmp_path / "blobs").mkdir()
    blob = tmp_path / "blobs" / f"sha256-{digest}"
    blob.write_bytes(content)
    (tmp_path / "manifest").write_bytes(manifest)
    lock = {"models": {"synthetic:only": {"manifest_path": "manifest", "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "blobs": {f"sha256:{digest}": len(content)}}}}
    assert verify(tmp_path, lock)["valid"] is True
    blob.write_bytes(b"X" + content[1:])
    assert verify(tmp_path, lock)["valid"] is False
    blob.write_bytes(content)
    (tmp_path / "manifest").write_bytes(b"wrong manifest")
    assert verify(tmp_path, lock)["valid"] is False
