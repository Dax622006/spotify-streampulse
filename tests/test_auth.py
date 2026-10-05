# spotify-streampulse/tests/test_auth.py
"""
Unit tests for Authentication and RBAC.
Covers T06.01–T06.07 and AC-AUTH-1..3.
"""
from __future__ import annotations
import json
import pytest

from streampulse.auth.credentials import hash_password, verify_password, load_users
from streampulse.auth.service import authenticate, reset_failures
from streampulse.auth.permissions import (
    ADMIN_PAGES,
    ANALYST_PAGES,
    ROLE_PAGES,
    is_page_permitted,
)
from streampulse.config import (
    ADMIN_ROLE,
    ANALYST_ROLE,
    LOCKOUT_FAILURES,
    USERS_FILE,
)


def test_password_hashing():
    salt, pwd_hash = hash_password("TestSecret123")
    assert len(salt) == 32  # 16 bytes hex
    assert len(pwd_hash) == 64  # sha256 hex
    assert verify_password("TestSecret123", salt, pwd_hash)
    assert not verify_password("WrongSecret", salt, pwd_hash)


def test_no_plaintext_passwords_in_users_json():
    """AC-AUTH-1: users.json contains only salted hashes."""
    users = load_users()
    assert "admin" in users
    assert "analyst" in users
    for user, data in users.items():
        assert "password" not in data
        assert "salt" in data
        assert "hash" in data
        assert len(data["salt"]) == 32
        assert len(data["hash"]) == 64


def test_auth_success():
    reset_failures("admin")
    reset_failures("analyst")
    ok, role, _ = authenticate("admin", "Admin@123")
    assert ok is True
    assert role == ADMIN_ROLE

    ok, role, _ = authenticate("analyst", "Analyst@123")
    assert ok is True
    assert role == ANALYST_ROLE


def test_auth_invalid_credentials():
    reset_failures("admin")
    ok, role, msg = authenticate("admin", "WrongPassword")
    assert ok is False
    assert role is None
    assert "Invalid" in msg

    ok, role, msg = authenticate("nonexistent_user", "AnyPassword")
    assert ok is False
    assert role is None
    assert "Invalid" in msg


def test_account_lockout():
    username = "admin"
    reset_failures(username)
    for _ in range(LOCKOUT_FAILURES):
        authenticate(username, "BadPassword")

    # 6th attempt must report locked out
    ok, role, msg = authenticate(username, "Admin@123")
    assert ok is False
    assert "locked" in msg.lower()

    # Reset for other tests
    reset_failures(username)


def test_permission_matrix():
    """AC-AUTH-2: Exact lists and no shared pages."""
    assert len(ADMIN_PAGES) == 6
    assert len(ANALYST_PAGES) == 8
    # Disjoint sets: no page ID shared
    assert set(ADMIN_PAGES).isdisjoint(set(ANALYST_PAGES))

    # All admin pages permitted for ADMIN, denied for ANALYST
    for page in ADMIN_PAGES:
        assert is_page_permitted(ADMIN_ROLE, page) is True
        assert is_page_permitted(ANALYST_ROLE, page) is False

    # All analyst pages permitted for ANALYST, denied for ADMIN
    for page in ANALYST_PAGES:
        assert is_page_permitted(ANALYST_ROLE, page) is True
        assert is_page_permitted(ADMIN_ROLE, page) is False

    # Unauthenticated / invalid role denied for all
    assert is_page_permitted(None, ADMIN_PAGES[0]) is False
    assert is_page_permitted("GUEST", ADMIN_PAGES[0]) is False
