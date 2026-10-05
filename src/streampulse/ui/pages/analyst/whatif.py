# spotify-streampulse/src/streampulse/ui/pages/analyst/whatif.py
"""
What-If Scenario Simulator: Interactive hypothetical inference using pre-trained pipelines.
R-WI-01..06: Read-only simulation, input contract enforcement, and multi-model outputs.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd

from streampulse.auth.guard import require_page
from streampulse.config import PAGE_WHATIF
from streampulse.ml.inference import InputSpec, predict_scenario, validate_whatif_input
from streampulse.ui.components.layout import page_header
from streampulse.ui.components.cache import get_analytics_frame

@require_page(PAGE_WHATIF)
def render():
    page_header(
        title="What-If Scenario Simulator",
        subtitle="Simulate hypothetical streaming conditions to forecast Subscription conversion and Revenue",
        concept="Prescriptive / What-If Analytics",
        icon="🔮",
    )

    df = get_analytics_frame()
    if df.empty:
        st.warning("⚠️ Warehouse contains no data.")
        return

    # Derive input specification from active data
    spec = InputSpec(
        streams_min=int(df["Streams"].min()),
        streams_max=int(df["Streams"].max()),
        duration_min=int(df["Duration_Minutes"].min()),
        duration_max=int(df["Duration_Minutes"].max()),
        devices=sorted(df["Device"].unique()),
        genres=sorted(df["Genre"].unique()),
        countries=sorted(df["Country"].unique()),
    )

    st.markdown("### 🎛️ Configure Simulation Parameters")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            streams_in = st.slider("Streams Count", min_value=spec.streams_min, max_value=spec.streams_max, value=10)
            duration_in = st.slider("Duration (Minutes)", min_value=spec.duration_min, max_value=spec.duration_max, value=30)
        with col2:
            device_in = st.selectbox("Device Type", options=spec.devices)
            genre_in = st.selectbox("Primary Genre", options=spec.genres)
        with col3:
            country_in = st.selectbox("Listener Country", options=spec.countries)
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("🚀 Run Scenario Simulation", type="primary", use_container_width=True)

    # Initialize scenario history in session
    if "scenario_history" not in st.session_state:
        st.session_state["scenario_history"] = []

    if run_btn:
        is_valid, errors, warnings = validate_whatif_input(
            streams_in, duration_in, device_in, genre_in, country_in, spec
        )

        for w in warnings:
            st.warning(f"⚠️ {w}")

        if not is_valid:
            for err in errors:
                st.error(f"❌ {err}")
        else:
            pred = predict_scenario(streams_in, duration_in, device_in, genre_in, country_in)

            st.markdown("### 🎯 Simulation Forecast Results")
            r1, r2 = st.columns(2)

            with r1:
                with st.container(border=True):
                    st.markdown("#### 💰 Revenue Projections")
                    c_slr, c_mlr = st.columns(2)
                    with c_slr:
                        st.metric("SLR Forecast", f"${pred.slr_revenue:.2f}")
                        st.caption("Simple Linear (Streams only)")
                    with c_mlr:
                        st.metric("MLR Forecast", f"${pred.mlr_revenue:.2f}")
                        st.caption("Multiple Linear (All features)")

            with r2:
                with st.container(border=True):
                    st.markdown("#### 👤 Subscription Tier Classification")
                    c_dt, c_nb = st.columns(2)
                    with c_dt:
                        st.metric("Decision Tree", pred.dt_subscription)
                        st.caption(f"Probabilities: {pred.dt_proba}")
                    with c_nb:
                        st.metric("Naive Bayes", pred.nb_subscription)
                        st.caption(f"Probabilities: {pred.nb_proba}")

            # Append to session history
            st.session_state["scenario_history"].append({
                "Streams": streams_in,
                "Duration": duration_in,
                "Device": device_in,
                "Genre": genre_in,
                "Country": country_in,
                "SLR Rev ($)": pred.slr_revenue,
                "MLR Rev ($)": pred.mlr_revenue,
                "DT Pred": pred.dt_subscription,
                "NB Pred": pred.nb_subscription,
            })

    if st.session_state["scenario_history"]:
        st.markdown("### 📜 Session Simulation History")
        st.dataframe(pd.DataFrame(st.session_state["scenario_history"]), use_container_width=True, hide_index=True)
        if st.button("Clear History"):
            st.session_state["scenario_history"] = []
            st.rerun()
