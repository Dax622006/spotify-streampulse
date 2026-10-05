# spotify-streampulse/src/streampulse/ui/pages/admin/data_quality.py
"""
Data Quality Monitor: Live audit of the 21 validation checks and anomaly quarantining.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd

from streampulse.auth.guard import require_page
from streampulse.config import PAGE_DATA_QUALITY
from streampulse.ingestion.validator import load_report
from streampulse.ui.components.layout import page_header

@require_page(PAGE_DATA_QUALITY)
def render():
    page_header(
        title="Data Quality Monitor",
        subtitle="Automated validation engine results, business rule verification, and anomaly quarantines",
        concept="Data Quality & Integrity",
        icon="🛡️",
    )

    report = load_report()
    if not report:
        st.warning("⚠️ No validation report found. Please run the data pipeline.")
        return

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Overall Quality Status", report.overall_status, delta="Operational")
    with m2:
        st.metric("Passed Checks", report.summary.get("passed", 0))
    with m3:
        st.metric("Quality Warnings", report.summary.get("warnings", 0))
    with m4:
        st.metric("Critical Failures", report.summary.get("failed", 0))

    st.markdown("### 📋 Audit Findings & Verification Rules")

    # Filter controls
    categories = sorted(list({c.category for c in report.checks}))
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_cat = st.selectbox("Filter by Category", options=["All"] + categories)
    with col_f2:
        selected_status = st.selectbox("Filter by Status", options=["All", "PASS", "WARN", "FAIL", "INFO"])

    # Table of checks
    rows = []
    for c in report.checks:
        if selected_cat != "All" and c.category != selected_cat:
            continue
        if selected_status != "All" and c.status != selected_status:
            continue
        rows.append({
            "ID": c.id,
            "Category": c.category,
            "Severity": c.severity,
            "Status": c.status,
            "Check Message": c.message,
            "Observed": str(c.observed)[:40],
            "Expected": str(c.expected)[:40],
        })

    df_report = pd.DataFrame(rows)
    st.dataframe(df_report, use_container_width=True, hide_index=True)

    st.markdown("### ⚠️ Quarantined Anomaly Log (Design Decision C1, C2)")
    with st.container(border=True):
        st.markdown(
            """
            - **`Song_ID` Quarantined (Rule V-K03):** The source dataset exhibits an anomaly where 50 distinct `Song_ID`s map to only 10 unique `Song_Name`s (10 names per ID). `Song_ID` is strictly prohibited from serving as a primary/natural key.
            - **`Artist_ID` / `Album_ID` Quarantined (Rules V-K04, V-K05):** Similarly, IDs do not exhibit 1:1 functional dependencies with entity names.
            - **Resolution Applied:** Dimension surrogate keys (`Track_SK`) are generated deterministically using clean `Song_Name` natural keys, preserving mathematical integrity throughout the star schema.
            """
        )
