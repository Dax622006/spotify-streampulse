# spotify-streampulse/src/streampulse/ui/pages/login.py
"""
Login page for Spotify StreamPulse.
R-RBAC-07, R-RBAC-08: Secure authentication with lockout protection.
"""
from __future__ import annotations
import streamlit as st

from streampulse.auth.permissions import get_default_page_for_role
from streampulse.auth.service import authenticate, is_locked_out
from streampulse.ui.components.layout import logo_html


def render_login_page() -> None:
    """Render the authentication login view."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 40px; margin-bottom: 24px;">
                <h1 style="margin-bottom: 0px;">{logo_html(48)}Spotify StreamPulse</h1>
                <p style="color: #e2e8f0; font-size: 15px;">Enterprise Data Warehousing & Predictive Mining</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.subheader("System Sign-In")

            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="e.g. admin or analyst").strip()
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

                if submitted:
                    if not username or not password:
                        st.error("Please provide both username and password.")
                    else:
                        success, role, message = authenticate(username, password)
                        if success and role:
                            st.session_state["authenticated"] = True
                            st.session_state["user"] = username
                            st.session_state["role"] = role
                            st.session_state["current_page"] = get_default_page_for_role(role)
                            st.success(f"Welcome, {username} ({role})!")
                            st.rerun()
                        else:
                            st.error(message)

        with st.expander("ℹ️ Demo Accounts (Academic Viva / Evaluation)"):
            st.markdown(
                """
                - **Administrator**: `admin` / `Admin@123` (Role: `ADMIN`)
                  *Access to: Data Warehouse, Star Schema DDL, Data Quality Monitor, System Logs, Model Status*
                - **Data Analyst**: `analyst` / `Analyst@123` (Role: `DATA_ANALYST`)
                  *Access to: OLAP Cube Operations, EDA, Regression, Classification, Clustering, Association Mining, What-If Simulator*
                """
            )
