# spotify-streampulse/src/streampulse/ui/pages/admin/data_warehouse.py
"""
Data Warehouse Inspector: Browse raw staging, dimensions, and fact tables.
"""
from __future__ import annotations
import streamlit as st

from streampulse.auth.guard import require_page
from streampulse.config import PAGE_DATA_WAREHOUSE
from streampulse.ui.components.layout import page_header
from streampulse.warehouse.repository import (
    get_dim_country,
    get_dim_device,
    get_dim_genre,
    get_dim_subscription,
    get_dim_time,
    get_dim_track,
    get_dim_user,
    get_fact_streams,
    get_stg_clean,
    get_stg_raw,
    get_table_row_counts,
)

@require_page(PAGE_DATA_WAREHOUSE)
def render():
    page_header(
        title="Data Warehouse Inspector",
        subtitle="Live inspection of Star Schema tables, surrogate keys, and staging areas",
        concept="Storage & Retrieval",
        icon="🏛️",
    )

    counts = get_table_row_counts()
    table_options = [
        "Fact_Streams",
        "Dim_User",
        "Dim_Track",
        "Dim_Genre",
        "Dim_Device",
        "Dim_Country",
        "Dim_Subscription",
        "Dim_Time",
        "Stg_Clean",
        "Stg_Raw",
    ]

    col_select, col_limit = st.columns([3, 1])
    with col_select:
        selected_table = st.selectbox("Select Warehouse Table", options=table_options)
    with col_limit:
        row_limit = st.slider("Display Limit", min_value=10, max_value=500, value=50, step=10)

    # Fetch selected table data
    table_fetchers = {
        "Fact_Streams": get_fact_streams,
        "Dim_User": get_dim_user,
        "Dim_Track": get_dim_track,
        "Dim_Genre": get_dim_genre,
        "Dim_Device": get_dim_device,
        "Dim_Country": get_dim_country,
        "Dim_Subscription": get_dim_subscription,
        "Dim_Time": get_dim_time,
        "Stg_Clean": get_stg_clean,
        "Stg_Raw": get_stg_raw,
    }

    df = table_fetchers[selected_table]()
    st.markdown(
        f"**Showing {min(len(df), row_limit)} of {counts.get(selected_table, len(df)):,} records** in `{selected_table}`"
    )
    st.dataframe(df.head(row_limit), use_container_width=True, hide_index=True)

    with st.expander(f"ℹ️ Technical Details & DDL Grain: {selected_table}"):
        st.markdown(f"- **Total Rows in DB:** `{len(df):,}`")
        st.markdown(f"- **Columns ({len(df.columns)}):** `{', '.join(df.columns)}`")
        if selected_table == "Fact_Streams":
            st.markdown("- **Grain:** 1 row per individual streaming event (`Stream_ID`).")
            st.markdown("- **Measures:** `Streams` (Additive), `Duration_Minutes` (Additive), `Revenue_USD` (Additive).")
            st.markdown("- **Foreign Keys:** `User_SK`, `Track_SK`, `Genre_SK`, `Device_SK`, `Country_SK`, `Subscription_SK`, `Date_SK`.")
        elif selected_table == "Dim_Track":
            st.markdown("- **Hierarchy:** `Artist_Name` -> `Album_Name` -> `Song_Name`.")
            st.markdown("- **Natural Key:** `Song_Name` (Source IDs quarantined due to documented quality anomalies).")
        elif selected_table == "Dim_Time":
            st.markdown("- **Key:** `Date_SK` (Smart integer key YYYYMMDD).")
            st.markdown("- **Coverage:** Full 365-day calendar year 2025 with `Is_Observed` flag.")
