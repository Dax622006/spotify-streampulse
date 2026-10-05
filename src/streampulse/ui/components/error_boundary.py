# spotify-streampulse/src/streampulse/ui/components/error_boundary.py
"""
Global error boundary per T16.04 and FR-INT-03.
"""
from __future__ import annotations
import logging
import traceback
from typing import Callable

import streamlit as st

from streampulse.logging_setup import get_logger, log_event

logger = get_logger(__name__)

def safe_render(page_fn: Callable, page_id: str) -> None:
    """Execute page render inside an error boundary."""
    try:
        page_fn()
    except Exception as exc:
        tb = traceback.format_exc()
        user = st.session_state.get("user", "unknown")
        role = st.session_state.get("role", "unknown")
        log_event(
            logger,
            logging.ERROR,
            "PAGE_CRASH",
            f"Unhandled exception on page '{page_id}': {exc}",
            user=user,
            role=role,
            exc_trace=tb[:1000],
        )
        st.error(f"⚠️ A system error occurred while rendering this page: **{type(exc).__name__}**")
        with st.expander("Show Diagnostic Details"):
            st.code(tb, language="python")
