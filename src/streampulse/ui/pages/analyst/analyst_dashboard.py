# spotify-streampulse/src/streampulse/ui/pages/analyst/analyst_dashboard.py
"""
Data Analyst Executive Dashboard: Business KPIs and streaming volume trends.
"""
from __future__ import annotations
import plotly.express as px
import streamlit as st
import pandas as pd

from streampulse.auth.guard import require_page
from streampulse.config import PAGE_ANALYST_DASHBOARD
from streampulse.ui.components.layout import page_header
from streampulse.ui.components.cache import get_analytics_frame

@require_page(PAGE_ANALYST_DASHBOARD)
def render():
    page_header(
        title="Streaming Business Analytics",
        subtitle="High-level streaming engagement, revenue generation, and demographic breakdowns",
        concept="Descriptive Business Intelligence",
        icon="📈",
    )

    df = get_analytics_frame()
    if df.empty:
        st.warning("⚠️ Warehouse contains no data. Please contact the administrator.")
        return

    # Top Metric KPIs
    total_streams = int(df["Streams"].sum())
    total_revenue = float(df["Revenue_USD"].sum())
    total_duration = int(df["Duration_Minutes"].sum())
    avg_streams = float(df["Streams"].mean())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Streams", f"{total_streams:,}")
    with k2:
        st.metric("Total Platform Revenue", f"${total_revenue:,.2f}")
    with k3:
        st.metric("Total Playback Hours", f"{total_duration / 60:,.1f} hrs")
    with k4:
        st.metric("Avg Streams / Session", f"{avg_streams:.2f}")

    st.markdown("### 📊 Engagement & Revenue Distributions")
    col1, col2 = st.columns(2)

    with col1:
        # Genre streams
        genre_summary = df.groupby("Genre", as_index=False)["Streams"].sum().sort_values(by="Streams", ascending=False)
        fig_genre = px.bar(
            genre_summary,
            x="Genre",
            y="Streams",
            title="Total Stream Volume by Genre",
            color="Streams",
            color_continuous_scale="Viridis",
        )
        fig_genre.update_layout(xaxis_title="Genre", yaxis_title="Streams", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_genre, use_container_width=True)

    with col2:
        # Device Breakdown
        device_summary = df.groupby(["Device", "Subscription"], as_index=False)["Streams"].sum()
        fig_device = px.bar(
            device_summary,
            x="Device",
            y="Streams",
            color="Subscription",
            barmode="group",
            title="Stream Volume by Device & Subscription Tier",
            color_discrete_map={"Free": "#64748b", "Premium": "#1DB954"},
        )
        fig_device.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_device, use_container_width=True)

    # Monthly Trend
    st.markdown("### 📅 Monthly Streaming Trajectory")
    monthly = df.groupby(["Year", "Month_Number", "Month_Name"], as_index=False)[["Streams", "Revenue_USD"]].sum()
    monthly.sort_values(by="Month_Number", inplace=True)

    fig_month = px.line(
        monthly,
        x="Month_Name",
        y="Streams",
        markers=True,
        title="Streaming Trajectory Across Calendar Year 2025",
        line_shape="spline",
    )
    fig_month.update_traces(line_color="#1DB954", line_width=3)
    fig_month.update_layout(xaxis_title="Month", yaxis_title="Streams", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_month, use_container_width=True)

    # Top Tracks
    st.markdown("### 🏆 Top 10 Tracks by Stream Count")
    top_tracks = (
        df.groupby(["Song_Name", "Artist_Name", "Album_Name"], as_index=False)[["Streams", "Revenue_USD"]]
        .sum()
        .sort_values(by="Streams", ascending=False)
    )
    st.dataframe(top_tracks, use_container_width=True, hide_index=True)
