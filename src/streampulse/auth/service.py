# spotify-streampulse/src/streampulse/auth/service.py
"""
Authentication service.
R-RBAC-08: Log LOGIN_SUCCESS and LOGIN_FAILURE; never log passwords.
Lockout policy: 5 consecutive failures locks the account for 30s.
"""
from __future__ import annotations
import logging
import time
from typing import Any

from streampulse.auth.credentials import load_users, verify_password
from streampulse.config import LOCKOUT_FAILURES, LOCKOUT_SECONDS
from streampulse.logging_setup import get_logger, log_event

logger = get_logger(__name__)

# In-memory failure tracking: {username: [timestamps_of_recent_failures]}
_failed_attempts: dict[str, list[float]] = {}


def is_locked_out(username: str) -> tuple[bool, int]:
    """Check if username is currently locked out.

    Returns:
        (is_locked, remaining_seconds)
    """
    now = time.time()
    history = _failed_attempts.get(username, [])
    # Keep only attempts within the lockout window
    recent = [t for t in history if now - t < LOCKOUT_SECONDS]
    _failed_attempts[username] = recent

    if len(recent) >= LOCKOUT_FAILURES:
        oldest_active = recent[-LOCKOUT_FAILURES]
        remaining = int(LOCKOUT_SECONDS - (now - oldest_active))
        return True, max(1, remaining)
    return False, 0


def authenticate(username: str, password: str) -> tuple[bool, str | None, str]:
    """Authenticate username and password.

    Returns:
        (success, role, message)
    """
    uname = username.strip()
    locked, remaining = is_locked_out(uname)
    if locked:
        msg = f"Account temporarily locked due to multiple failed attempts. Try again in {remaining}s."
        log_event(
            logger,
            logging.WARNING,
            "LOGIN_LOCKOUT",
            f"Login attempt on locked account '{uname}'",
            user=uname,
            remaining_seconds=remaining,
        )
        return False, None, msg

    users = load_users()
    user_record = users.get(uname)

    if not user_record:
        # User not found; simulate constant-time work
        verify_password("dummy", "00" * 16, "00" * 32)
        _record_failure(uname)
        log_event(
            logger,
            logging.WARNING,
            "LOGIN_FAILURE",
            f"Login failed for non-existent user '{uname}'",
            user=uname,
        )
        return False, None, "Invalid username or password."

    salt = user_record.get("salt", "")
    pwd_hash = user_record.get("hash", "")
    role = user_record.get("role")

    if not verify_password(password, salt, pwd_hash):
        _record_failure(uname)
        log_event(
            logger,
            logging.WARNING,
            "LOGIN_FAILURE",
            f"Invalid password for user '{uname}'",
            user=uname,
        )
        return False, None, "Invalid username or password."

    # Login succeeded
    _failed_attempts.pop(uname, None)
    log_event(
        logger,
        logging.INFO,
        "LOGIN_SUCCESS",
        f"User '{uname}' logged in successfully with role '{role}'",
        user=uname,
        role=role,
    )
    return True, role, "Login successful."


def _record_failure(username: str) -> None:
    now = time.time()
    history = _failed_attempts.setdefault(username, [])
    history.append(now)


def reset_failures(username: str) -> None:
    """Helper for testing or manual admin unlock."""
    _failed_attempts.pop(username, None)
