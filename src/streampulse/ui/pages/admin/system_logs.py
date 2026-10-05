# spotify-streampulse/src/streampulse/ui/pages/admin/system_logs.py
"""
System Logs Viewer: Search and filter rotating structured JSON audit trail.
"""
from __future__ import annotations
import json
import streamlit as st
import pandas as pd

from streampulse.auth.guard import require_page
from streampulse.config import LOG_FILE, PAGE_SYSTEM_LOGS
from streampulse.ui.components.layout import page_header

@require_page(PAGE_SYSTEM_LOGS)
def render():
    page_header(
        title="System Audit & Event Logs",
        subtitle="Rotating structured JSON telemetry, security events, and execution trace",
        concept="Auditing & Observability",
        icon="📜",
    )

    if not LOG_FILE.exists():
        st.info("Log file has not been initialized yet.")
        return

    # Read log entries
    lines = LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except Exception:
            continue

    # Reverse chronologically
    entries.reverse()

    st.markdown(f"**Total Captured Log Records:** `{len(entries):,}`")

    # Filters
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        level_filter = st.selectbox("Log Level", options=["ALL", "INFO", "WARNING", "ERROR", "DEBUG"])
    with col2:
        event_filter = st.selectbox(
            "Event Filter",
            options=["ALL"] + sorted(list({e.get("event") for e in entries if e.get("event")})),
        )
    with col3:
        search_kw = st.text_input("Search Message Text", placeholder="e.g. login, pipeline, sha256").strip().lower()

    filtered = []
    for e in entries:
        lvl = e.get("level", "INFO")
        ev = e.get("event", "")
        msg = e.get("message", "")

        if level_filter != "ALL" and lvl != level_filter:
            continue
        if event_filter != "ALL" and ev != event_filter:
            continue
        if search_kw and (search_kw not in msg.lower() and search_kw not in ev.lower()):
            continue

        filtered.append({
            "Timestamp": e.get("ts", "")[:19],
            "Level": lvl,
            "Event": ev,
            "User": e.get("user", "-"),
            "Module": e.get("module", ""),
            "Message": msg,
        })

    st.dataframe(pd.DataFrame(filtered), use_container_width=True, hide_index=True)
