from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from app.capsules.models import CapsuleSignature


class CapsuleSigner(ABC):
    algorithm: str
    key_id: str

    @abstractmethod
    def sign(self, root_hash: str) -> CapsuleSignature:
        raise NotImplementedError

    @abstractmethod
    def verify(self, root_hash: str, signature: CapsuleSignature) -> bool:
        raise NotImplementedError


class Ed25519CapsuleSigner(CapsuleSigner):
    algorithm = "Ed25519"

    def __init__(self, key_id: str, private_key: Ed25519PrivateKey | None, public_key: Ed25519PublicKey) -> None:
        self.key_id = key_id
        self._private_key = private_key
        self._public_key = public_key

    @classmethod
    def generate_for_testing(cls, key_id: str = "test-only") -> "Ed25519CapsuleSigner":
        private = Ed25519PrivateKey.generate()
        return cls(key_id, private, private.public_key())

    @classmethod
    def from_public_bytes(cls, key_id: str, value: bytes) -> "Ed25519CapsuleSigner":
        return cls(key_id, None, Ed25519PublicKey.from_public_bytes(value))

    def public_bytes(self) -> bytes:
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def sign(self, root_hash: str) -> CapsuleSignature:
        if self._private_key is None:
            raise ValueError("This signer has no private key")
        value = self._private_key.sign(root_hash.encode("ascii"))
        return CapsuleSignature(
            algorithm=self.algorithm,
            key_id=self.key_id,
            signed_root_hash=root_hash,
            signature=base64.b64encode(value).decode("ascii"),
        )

    def verify(self, root_hash: str, signature: CapsuleSignature) -> bool:
        if signature.algorithm != self.algorithm or signature.key_id != self.key_id:
            return False
        try:
            self._public_key.verify(base64.b64decode(signature.signature, validate=True), root_hash.encode("ascii"))
            return signature.signed_root_hash == root_hash
        except (InvalidSignature, ValueError):
            return False


class WorkcellTrustStore:
    """Local public-key trust material only; it never stores private keys."""

    def __init__(self) -> None:
        self._keys: dict[str, Ed25519CapsuleSigner] = {}

    def add_ed25519(self, key_id: str, public_key: bytes) -> None:
        self._keys[key_id] = Ed25519CapsuleSigner.from_public_bytes(key_id, public_key)

    def load_ed25519_file(self, key_id: str, path: Path) -> None:
        self.add_ed25519(key_id, path.read_bytes())

    def get(self, key_id: str) -> Ed25519CapsuleSigner | None:
        return self._keys.get(key_id)
