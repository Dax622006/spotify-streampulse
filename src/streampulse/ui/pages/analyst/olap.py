# spotify-streampulse/src/streampulse/ui/pages/analyst/olap.py
"""
OLAP Cube Operations Console: Interactive Slice, Dice, Roll-up, Drill-down, and Pivot.
R-OLAP-01..06: Full multidimensional OLAP analytical engine.
"""
from __future__ import annotations
import plotly.express as px
import streamlit as st
import pandas as pd

from streampulse.auth.guard import require_page
from streampulse.config import (
    ADDITIVE_MEASURES,
    PAGE_OLAP,
    TIME_HIERARCHY,
    TRACK_HIERARCHY,
)
from streampulse.ui.components.layout import page_header
from streampulse.ui.components.cache import get_analytics_frame, olap_aggregate

@require_page(PAGE_OLAP)
def render():
    page_header(
        title="OLAP Multidimensional Analysis",
        subtitle="Execute Slice, Dice, Roll-up, Drill-down, and Pivot operations on the Star Schema Cube",
        concept="OLAP Operations",
        icon="🧊",
    )

    df_full = get_analytics_frame()
    if df_full.empty:
        st.warning("⚠️ Warehouse is empty.")
        return

    # Tabs for OLAP Operations
    tab_pivot, tab_slice, tab_dice, tab_hierarchy = st.tabs([
        "🔄 Pivot & Aggregate",
        "🔪 Slice (Single Dimension)",
        "🎲 Dice (Sub-Cube Filtering)",
        "🧗 Roll-up & Drill-down",
    ])

    dimension_cols = [
        "Genre",
        "Device",
        "Country",
        "Subscription",
        "Artist_Name",
        "Album_Name",
        "Song_Name",
        "Year",
        "Quarter",
        "Month_Name",
    ]

    with tab_pivot:
        st.markdown("#### Dynamic Multi-Axis Pivot")
        col_r, col_c, col_m, col_a = st.columns(4)
        with col_r:
            row_dim = st.selectbox("Row Dimension", options=dimension_cols, index=0)
        with col_c:
            col_dim = st.selectbox(
                "Column Dimension (Optional)",
                options=["None"] + [d for d in dimension_cols if d != row_dim],
                index=1,
            )
        with col_m:
            measure = st.selectbox("Measure", options=ADDITIVE_MEASURES, index=0)
        with col_a:
            agg_func = st.selectbox("Aggregation", options=["SUM", "AVG", "COUNT", "MIN", "MAX"], index=0)

        cols_list = [] if col_dim == "None" else [col_dim]
        res = olap_aggregate(rows=[row_dim], cols=cols_list, measure=measure, agg=agg_func)

        st.markdown(f"**Result Cube ({len(res)} rows):**")
        st.dataframe(res, use_container_width=True)

        if col_dim != "None" and not res.empty:
            # Heatmap visualization
            pivot_data = res.set_index(row_dim)
            fig = px.imshow(
                pivot_data,
                text_auto=True,
                aspect="auto",
                title=f"{agg_func}({measure}) across {row_dim} and {col_dim}",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_slice:
        st.markdown("#### Slice: Fix one dimension to a single value")
        slice_dim = st.selectbox("Slice Dimension", options=["Country", "Device", "Genre", "Subscription"], key="sl_dim")
        unique_vals = sorted(df_full[slice_dim].unique())
        slice_val = st.selectbox("Filter Value", options=unique_vals, key="sl_val")

        sliced_df = olap_aggregate(
            rows=["Genre" if slice_dim != "Genre" else "Device"],
            cols=[],
            measure="Streams",
            agg="SUM",
            filters={slice_dim: [slice_val]},
        )
        st.markdown(f"**Cube Slice where `{slice_dim} = '{slice_val}'`:**")
        st.dataframe(sliced_df, use_container_width=True)

    with tab_dice:
        st.markdown("#### Dice: Filter across multiple dimensions simultaneously")
        c1, c2 = st.columns(2)
        with c1:
            sel_genres = st.multiselect("Select Genres", options=sorted(df_full["Genre"].unique()), default=["Pop", "Rock"])
        with c2:
            sel_devices = st.multiselect("Select Devices", options=sorted(df_full["Device"].unique()), default=["Mobile", "Web"])

        if sel_genres and sel_devices:
            diced_df = olap_aggregate(
                rows=["Genre"],
                cols=["Device"],
                measure="Revenue_USD",
                agg="SUM",
                filters={"Genre": sel_genres, "Device": sel_devices},
            )
            st.markdown(f"**Sub-Cube Revenue Aggregate (${diced_df.iloc[:, 1:].sum().sum():,.2f} total):**")
            st.dataframe(diced_df, use_container_width=True)
        else:
            st.info("Select at least one genre and device to generate sub-cube.")

    with tab_hierarchy:
        st.markdown("#### Roll-up & Drill-down Along Predefined Hierarchies")
        h_choice = st.radio("Select Hierarchy", options=["Time Hierarchy (Date -> Month -> Quarter -> Year)", "Track Hierarchy (Song -> Album -> Artist)"])

        if "Time" in h_choice:
            level = st.select_slider("Aggregation Level", options=["Date", "Month", "Quarter", "Year"], value="Quarter")
            level_col_map = {"Date": "Date", "Month": "Month_Name", "Quarter": "Quarter", "Year": "Year"}
            hier_col = level_col_map[level]
        else:
            level = st.select_slider("Aggregation Level", options=["Song", "Album", "Artist"], value="Album")
            level_col_map = {"Song": "Song_Name", "Album": "Album_Name", "Artist": "Artist_Name"}
            hier_col = level_col_map[level]

        hier_df = olap_aggregate(rows=[hier_col], cols=[], measure="Streams", agg="SUM")
        st.markdown(f"**Aggregated at `{level}` level ({len(hier_df)} members):**")
        st.dataframe(hier_df, use_container_width=True)

        fig_h = px.bar(hier_df, x=hier_col, y="Streams", color="Streams", title=f"Streams Aggregated by {level}")
        st.plotly_chart(fig_h, use_container_width=True)
