from __future__ import annotations

import base64
import hashlib
import hmac
import os


class PasswordHasher:
    """Versioned salted PBKDF2-HMAC-SHA256 password hashes."""

    algorithm = "pbkdf2-sha256"

    def __init__(self, iterations: int = 210_000) -> None:
        self.iterations = max(100_000, iterations)

    def hash(self, password: str) -> str:
        if len(password) < 10:
            raise ValueError("Password must contain at least 10 characters")
        salt = os.urandom(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, self.iterations)
        return "$".join((
            self.algorithm, str(self.iterations),
            base64.urlsafe_b64encode(salt).decode("ascii"),
            base64.urlsafe_b64encode(digest).decode("ascii"),
        ))

    def verify(self, password: str, encoded: str) -> bool:
        try:
            algorithm, iterations, salt_value, digest_value = encoded.split("$", 3)
            if algorithm != self.algorithm:
                return False
            salt = base64.urlsafe_b64decode(salt_value.encode("ascii"))
            expected = base64.urlsafe_b64decode(digest_value.encode("ascii"))
            actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
            return hmac.compare_digest(actual, expected)
        except (ValueError, TypeError):
            return False
