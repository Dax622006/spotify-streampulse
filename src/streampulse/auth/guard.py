# spotify-streampulse/src/streampulse/auth/guard.py
"""
Page authorization guard.
R-RBAC-04: Server-side authorization check before rendering any page.
R-RBAC-05: Denials logged as ACCESS_DENIED.
AC-AUTH-3: Cross-role requests are denied and logged.
"""
from __future__ import annotations
import logging
from typing import Callable

import streamlit as st

from streampulse.auth.permissions import is_page_permitted
from streampulse.logging_setup import get_logger, log_event

logger = get_logger(__name__)


def check_page_access(page_id: str) -> bool:
    """Verify currently authenticated user has permission to view page_id.

    Returns:
        True if allowed to proceed.
        False if denied (and renders error UI + logs violation).
    """
    user = st.session_state.get("user")
    role = st.session_state.get("role")
    is_auth = st.session_state.get("authenticated", False)

    if not is_auth or not user or not role:
        log_event(
            logger,
            logging.WARNING,
            "UNAUTHENTICATED_ACCESS",
            f"Unauthenticated attempt to access page '{page_id}'",
            user="anonymous",
            page_id=page_id,
        )
        st.error("⛔ Authentication Required. Please log in to continue.")
        return False

    if not is_page_permitted(role, page_id):
        log_event(
            logger,
            logging.WARNING,
            "ACCESS_DENIED",
            f"User '{user}' with role '{role}' denied access to page '{page_id}'",
            user=user,
            role=role,
            page_id=page_id,
        )
        st.error(
            f"⛔ Access Denied. Your role ({role}) does not have permission to access this page."
        )
        return False

    return True


def require_page(page_id: str) -> Callable:
    """Decorator for page render functions to enforce server-side RBAC."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if check_page_access(page_id):
                return func(*args, **kwargs)
            return None
        return wrapper
    return decorator
