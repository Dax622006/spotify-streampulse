# spotify-streampulse/src/streampulse/ui/navigation.py
"""
Dynamic navigation sidebar based on authenticated role.
R-RBAC-02: Generates role-specific menu (Admin=6, Analyst=8 items).
"""
from __future__ import annotations
import streamlit as st

from streampulse.ui.components.layout import logo_html

from streampulse.auth.permissions import (
    PAGE_ICONS,
    PAGE_LABELS,
    get_default_page_for_role,
    get_pages_for_role,
)
from streampulse.logging_setup import get_logger, log_event

logger = get_logger(__name__)


def render_sidebar() -> str:
    """Render the sidebar with user info, navigation menu, and logout button.

    Returns:
        Selected page_id.
    """
    user = st.session_state.get("user", "")
    role = st.session_state.get("role", "")

    permitted_pages = get_pages_for_role(role)
    if not permitted_pages:
        return ""

    with st.sidebar:
        st.markdown(f"<h3>{logo_html(30)}Spotify StreamPulse</h3>", unsafe_allow_html=True)
        st.caption("Data Warehousing & Mining System")
        st.divider()

        # User Profile Badge
        role_color = "#3b82f6" if role == "ADMIN" else "#10b981"
        st.markdown(
            f"""
            <div style="background: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 16px;">
                <div style="font-weight: 600; font-size: 14px; margin-bottom: 4px;">👤 {user}</div>
                <div style="display: inline-block; background: {role_color}; color: white; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 12px;">{role}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Navigation")

        # Determine current active page
        current_page = st.session_state.get("current_page")
        if current_page not in permitted_pages:
            current_page = permitted_pages[0]
            st.session_state["current_page"] = current_page

        # Render navigation options
        options = [
            f"{PAGE_ICONS.get(p, '📄')} {PAGE_LABELS.get(p, p)}" for p in permitted_pages
        ]
        # The radio is driven by its own key. (Passing a changing `index=` made
        # Streamlit treat it as a brand-new widget after each click, so every
        # second click was silently ignored.)
        if st.session_state.get("nav_choice") not in options:
            st.session_state["nav_choice"] = options[permitted_pages.index(current_page)]

        selected_label = st.radio(
            "Select View",
            options=options,
            key="nav_choice",
            label_visibility="collapsed",
        )

        selected_page = permitted_pages[options.index(selected_label)]
        st.session_state["current_page"] = selected_page

        st.divider()
        if st.button("🚪 Log Out", use_container_width=True):
            log_event(logger, 20, "LOGOUT", f"User '{user}' logged out", user=user, role=role)
            for k in ["authenticated", "user", "role", "current_page", "nav_choice"]:
                st.session_state.pop(k, None)
            st.rerun()

    return selected_page
