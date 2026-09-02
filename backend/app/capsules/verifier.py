from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from app.capsules.models import (
    CapsuleManifest,
    CapsuleSignature,
    CapsuleStatus,
    CapsuleVerificationFailure,
    CapsuleVerificationResult,
    SignatureStatus,
)
from app.capsules.signing import WorkcellTrustStore
from app.identity import ContentIdentityService


class EvidenceCapsuleVerifier:
    META_FILES = {"capsule_manifest.json", "hashes.sha256", "signature.json"}

    def __init__(self, trust_store: WorkcellTrustStore | None = None, *, unsigned_allowed: bool = True) -> None:
        self.trust_store = trust_store or WorkcellTrustStore()
        self.unsigned_allowed = unsigned_allowed
        self.identity = ContentIdentityService()

    @staticmethod
    def _failure(type_: str, message: str, path: str | None = None) -> CapsuleVerificationFailure:
        return CapsuleVerificationFailure(type=type_, message=message, path=path)

    def verify(self, root: Path) -> CapsuleVerificationResult:
        root = root.resolve()
        failures: list[CapsuleVerificationFailure] = []
        manifest_path = root / "capsule_manifest.json"
        try:
            manifest = CapsuleManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            manifest_valid = True
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            return CapsuleVerificationResult(
                status=CapsuleStatus.invalid, manifest_valid=False, hashes_valid=False,
                root_hash_valid=False, signature_status=SignatureStatus.invalid,
                failures=[self._failure("CAPSULE_SCHEMA_INVALID", str(exc), "capsule_manifest.json")],
            )
        expected = {item.path: item for item in manifest.files}
        actual_payload = {
            path.relative_to(root).as_posix() for path in root.rglob("*")
            if path.is_file() and path.name not in self.META_FILES
        }
        for missing in sorted(set(expected) - actual_payload):
            failures.append(self._failure("MISSING_FILE", "Manifest file is missing", missing))
        for extra in sorted(actual_payload - set(expected)):
            failures.append(self._failure("UNEXPECTED_FILE", "File is not declared in capsule manifest", extra))
        artifact_count = sum(path.startswith("artifacts/") for path in expected)
        artifact_valid = 0
        actual_hashes: dict[str, str] = {}
        for relative, item in sorted(expected.items()):
            path = (root / relative).resolve()
            if root not in path.parents or not path.is_file():
                continue
            digest = self.identity.hash_file(path)
            actual_hashes[relative] = digest
            if digest != item.sha256 or path.stat().st_size != item.size:
                failures.append(self._failure("CAPSULE_HASH_MISMATCH", "Stored file identity does not match", relative))
            elif relative.startswith("artifacts/"):
                artifact_valid += 1
        hashes_valid = not any(item.type in {"MISSING_FILE", "UNEXPECTED_FILE", "CAPSULE_HASH_MISMATCH"} for item in failures)
        for artifact in manifest.artifacts:
            relative = str(artifact.get("path", ""))
            declared_hash = str(artifact.get("sha256", ""))
            file_identity = expected.get(relative)
            if not file_identity or file_identity.sha256 != declared_hash:
                failures.append(self._failure("ARTIFACT_IDENTITY_MISMATCH", "Artifact metadata does not match its capsule file identity", relative or None))
                hashes_valid = False
        workcell_files_path = root / "execution" / "workcell_file_manifest.json"
        try:
            workcell_files = json.loads(workcell_files_path.read_text(encoding="utf-8"))
            calculated_workcell_hash = self.identity.hash_directory_manifest(workcell_files)
            if calculated_workcell_hash != manifest.workcell.hash:
                failures.append(self._failure("WORKCELL_IDENTITY_MISMATCH", "Workcell file manifest does not match the declared Workcell hash", "execution/workcell_file_manifest.json"))
                hashes_valid = False
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            failures.append(self._failure("WORKCELL_IDENTITY_MISMATCH", str(exc), "execution/workcell_file_manifest.json"))
            hashes_valid = False
        calculated_root = self.identity.hash_directory_manifest({path: item.sha256 for path, item in expected.items()})
        root_valid = calculated_root == manifest.capsule_root_hash
        if not root_valid:
            failures.append(self._failure("CAPSULE_ROOT_HASH_MISMATCH", "Capsule root identity is inconsistent", "capsule_manifest.json"))
        hashes_file = root / "hashes.sha256"
        expected_lines = "".join(f"{item.sha256}  {item.path}\n" for item in sorted(manifest.files, key=lambda value: value.path))
        if not hashes_file.is_file() or hashes_file.read_text(encoding="utf-8") != expected_lines:
            failures.append(self._failure("HASH_MANIFEST_INVALID", "hashes.sha256 is missing or inconsistent", "hashes.sha256"))
            hashes_valid = False
        signature_path = root / "signature.json"
        signature_valid: bool | None = None
        if not signature_path.exists():
            signature_status = SignatureStatus.unsigned
            if not self.unsigned_allowed:
                failures.append(self._failure("CAPSULE_SIGNATURE_INVALID", "Unsigned capsules are rejected by strict policy"))
        else:
            try:
                signature = CapsuleSignature.model_validate_json(signature_path.read_text(encoding="utf-8"))
                signer = self.trust_store.get(signature.key_id)
                if signer is None:
                    signature_status = SignatureStatus.signed_unverified
                else:
                    signature_valid = signer.verify(manifest.capsule_root_hash, signature)
                    signature_status = SignatureStatus.valid if signature_valid else SignatureStatus.invalid
                    if not signature_valid:
                        failures.append(self._failure("CAPSULE_SIGNATURE_INVALID", "Signature verification failed", "signature.json"))
            except (OSError, ValidationError, json.JSONDecodeError) as exc:
                signature_status = SignatureStatus.invalid
                signature_valid = False
                failures.append(self._failure("CAPSULE_SIGNATURE_INVALID", str(exc), "signature.json"))
        valid = manifest_valid and hashes_valid and root_valid and not failures
        return CapsuleVerificationResult(
            status=CapsuleStatus.valid if valid else CapsuleStatus.invalid,
            manifest_valid=manifest_valid, hashes_valid=hashes_valid,
            root_hash_valid=root_valid, signature_status=signature_status,
            signature_valid=signature_valid, artifact_count=artifact_count,
            artifact_valid_count=artifact_valid, failures=failures,
        )
