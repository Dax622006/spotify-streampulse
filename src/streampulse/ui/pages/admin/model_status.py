# spotify-streampulse/src/streampulse/ui/pages/admin/model_status.py
"""
Model Status Console: Registry inspection, metric baselines, and staleness audit.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd

from streampulse.auth.guard import require_page
from streampulse.config import PAGE_MODEL_STATUS, SOURCE_FILE
from streampulse.ingestion.loader import sha256_of
from streampulse.ml.registry import registry
from streampulse.orchestration import run_all
from streampulse.ui.components.layout import page_header

@require_page(PAGE_MODEL_STATUS)
def render():
    page_header(
        title="Machine Learning Model Registry",
        subtitle="Catalog of deployed models, training feature sets, staleness checks, and evaluation baselines",
        concept="Model Governance & MLOps",
        icon="🤖",
    )

    current_sha = sha256_of(SOURCE_FILE)
    models = registry.list_models()

    if not models:
        st.warning("⚠️ No registered models found. Run the pipeline to train and register models.")
        return

    # Trigger Retraining Button
    col_stat, col_btn = st.columns([3, 1])
    with col_stat:
        st.markdown(f"**Current Dataset Hash:** `{current_sha[:20]}...`")
    with col_btn:
        if st.button("🔄 Retrain All Models", type="primary", use_container_width=True):
            with st.spinner("Retraining full ML pipeline..."):
                run_all()
            st.success("All models successfully retrained and updated in registry!")
            st.rerun()

    st.markdown("### 📦 Registered Model Portfolio")
    for m in models:
        is_stale = registry.is_stale(m.model_id, current_sha)
        stale_badge = "🔴 Stale (Source Changed)" if is_stale else "🟢 Fresh (Synced)"

        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.markdown(f"#### {m.model_name} (`{m.model_id}`)")
                st.caption(f"Type: **{m.model_type.upper()}** | Trained: `{m.trained_at[:19]}`")
            with c2:
                st.markdown(f"**Status:** {stale_badge}")
                st.markdown(f"**Target Variable:** `{m.target or 'None (Unsupervised)'}`")
            with c3:
                st.markdown(f"**Features ({len(m.features)}):** `{', '.join(m.features)}`")

            # Metrics row
            st.markdown("**Evaluated Metrics:**")
            st.json(m.metrics)
