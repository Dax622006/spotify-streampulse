# spotify-streampulse/src/streampulse/ui/pages/admin/star_schema.py
"""
Star Schema Architectural Blueprint and Interactive DDL Viewer.
"""
from __future__ import annotations
import streamlit as st

from streampulse.auth.guard import require_page
from streampulse.config import PAGE_STAR_SCHEMA
from streampulse.ui.components.layout import page_header
from streampulse.warehouse.builder import DDL_STATEMENTS

@require_page(PAGE_STAR_SCHEMA)
def render():
    page_header(
        title="Star Schema Architecture",
        subtitle="Dimensional modeling blueprint, ER relationships, and DDL specifications",
        concept="Dimensional Modeling",
        icon="⭐",
    )

    st.markdown("### 🗺️ Dimensional Topology (Star Schema)")
    st.markdown(
        """
        ```
                            +-------------------+
                            |     Dim_User      |
                            |-------------------|
                            | User_SK (PK)      |
                            | User_ID (NK)      |
                            | User_Name         |
                            +---------+---------+
                                      | 1:N
        +-------------------+         |         +-------------------+
        |     Dim_Track     |         |         |     Dim_Genre     |
        |-------------------|         |         |-------------------|
        | Track_SK (PK)     +-----+   |   +-----+ Genre_SK (PK)     |
        | Song_Name (NK)    | 1:N |   |   | 1:N | Genre_Name (NK)   |
        | Artist_Name       |     v   v   v     |                   |
        | Album_Name        |  +-------------+  +-------------------+
        +-------------------+  | Fact_Streams|
                               |-------------|
        +-------------------+  | Stream_ID*  |  +-------------------+
        |    Dim_Device     |  | User_SK     |  |    Dim_Country    |
        |-------------------|  | Track_SK    |  |-------------------|
        | Device_SK (PK)    +->| Genre_SK    |<-+ Country_SK (PK)   |
        | Device_Name (NK)  |  | Device_SK   |  | Country_Name (NK) |
        +-------------------+  | Country_SK  |  +-------------------+
                               | Sub_SK      |
        +-------------------+  | Date_SK     |  +-------------------+
        | Dim_Subscription  +->| Streams     |<-+     Dim_Time      |
        |-------------------|  | Duration    |  |-------------------|
        | Sub_SK (PK)       |  | Revenue_USD |  | Date_SK (PK)      |
        | Sub_Name (NK)     |  +-------------+  | Full_Date, Year...|
        +-------------------+                   +-------------------+
        * Stream_ID is a degenerate dimension key (transaction identifier).
        ```
        """
    )

    st.markdown("### 📋 Dimensional Design Specifications")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Fact Table Characteristics")
            st.markdown("- **Name:** `Fact_Streams`")
            st.markdown("- **Grain:** One physical row per streaming session (`Stream_ID`).")
            st.markdown("- **Measures:**")
            st.markdown("  - `Streams`: Integer count of track plays (Fully additive).")
            st.markdown("  - `Duration_Minutes`: Total playback time (Fully additive).")
            st.markdown("  - `Revenue_USD`: Platform earnings (Fully additive).")
            st.markdown("- **Degenerate Dimension:** `Stream_ID` preserved on fact record.")

    with col2:
        with st.container(border=True):
            st.markdown("#### Dimension Characteristics")
            st.markdown("- **Surrogate Keys:** Autoincrementing integers (1..N).")
            st.markdown("- **Smart Key:** `Dim_Time.Date_SK` formatted as `YYYYMMDD` integer.")
            st.markdown("- **Natural Key Constraints:** Unique natural keys on each dimension.")
            st.markdown("- **Integrity:** `PRAGMA foreign_keys = ON` enforced at SQLite engine level.")

    with st.expander("📜 Full Warehouse DDL Specification (Executable SQL)"):
        st.code(DDL_STATEMENTS, language="sql")
