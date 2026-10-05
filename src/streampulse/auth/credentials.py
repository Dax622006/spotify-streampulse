# spotify-streampulse/src/streampulse/auth/credentials.py
"""
Credential store and password hashing.
R-RBAC-07: PBKDF2-HMAC-SHA256 (>= 200,000 iterations), random 16-byte salt.
Plaintext passwords are never stored or logged.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

from streampulse.config import PBKDF2_ITERATIONS, USERS_FILE
from streampulse.logging_setup import get_logger

logger = get_logger(__name__)


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """Hash password using PBKDF2-HMAC-SHA256.

    Returns:
        (salt_hex, hash_hex)
    """
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return salt.hex(), dk.hex()


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    """Verify password against salt and hash using constant-time comparison."""
    try:
        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)
        actual_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            PBKDF2_ITERATIONS,
        )
        return hmac.compare_digest(actual_hash, expected_hash)
    except Exception:
        return False


def load_users() -> dict[str, dict[str, Any]]:
    """Load users from config/users.json. Creates seed users if file is missing."""
    if not USERS_FILE.exists():
        ensure_seed_users()
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error(f"Failed to read users file: {exc}")
        return {}


def save_users(users: dict[str, dict[str, Any]]) -> None:
    """Save user store to config/users.json."""
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def ensure_seed_users() -> None:
    """Ensure config/users.json exists with admin and analyst accounts."""
    if USERS_FILE.exists():
        return

    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Default initial passwords per Design.md: Admin@123 and Analyst@123
    salt_admin, hash_admin = hash_password("Admin@123")
    salt_analyst, hash_analyst = hash_password("Analyst@123")

    seed_data = {
        "admin": {
            "role": "ADMIN",
            "salt": salt_admin,
            "hash": hash_admin,
        },
        "analyst": {
            "role": "DATA_ANALYST",
            "salt": salt_analyst,
            "hash": hash_analyst,
        },
    }
    save_users(seed_data)
    logger.info("Initialized default users store in config/users.json")
