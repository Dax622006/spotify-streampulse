# spotify-streampulse/src/streampulse/ui/pages/analyst/association.py
"""
Association Rule Mining: Apriori algorithm over User-Genre listening baskets.
R-ML-27..28: Market basket analysis, support/confidence/lift sliders, and rule scatter plots.
"""
from __future__ import annotations
import plotly.express as px
import streamlit as st
import pandas as pd

from streampulse.auth.guard import require_page
from streampulse.config import PAGE_ASSOCIATION
from streampulse.ui.components.cache import run_apriori_pipeline
from streampulse.ui.components.layout import page_header
from streampulse.ui.components.cache import get_analytics_frame

@require_page(PAGE_ASSOCIATION)
def render():
    page_header(
        title="Genre Affinity Association Mining",
        subtitle="Discover co-listening patterns and genre affinities using the Apriori algorithm",
        concept="Association Rule Mining",
        icon="🛒",
    )

    df = get_analytics_frame()
    if df.empty:
        st.warning("⚠️ Warehouse contains no data.")
        return

    # Controls
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        min_supp = st.slider("Minimum Support", min_value=0.10, max_value=0.80, value=0.40, step=0.05)
    with c2:
        min_conf = st.slider("Minimum Confidence", min_value=0.20, max_value=1.00, value=0.60, step=0.05)
    with c3:
        max_length = st.slider("Max Itemset Length", min_value=2, max_value=4, value=3)
    with c4:
        min_lift_val = st.slider("Minimum Lift", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

    result = run_apriori_pipeline(
        df,
        min_support=min_supp,
        min_confidence=min_conf,
        max_len=max_length,
        min_lift=min_lift_val,
    )

    # Basket Summary
    st.markdown("#### 🧺 Transaction Basket Summary (1 User = 1 Basket)")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total Baskets (Users)", result.summary["num_transactions"])
    with s2:
        st.metric("Catalog Genres", result.summary["num_items"])
    with s3:
        st.metric("Mean Basket Size", f"{result.summary['basket_size_mean']:.1f} genres")
    with s4:
        st.metric("Discovered Rules", len(result.rules))

    tab_rules, tab_items, tab_scatter = st.tabs([
        "📜 Association Rules",
        "📦 Frequent Itemsets",
        "📊 Support-Confidence-Lift Plot",
    ])

    with tab_rules:
        if not result.rules.empty:
            st.markdown(f"**Found {len(result.rules)} rules meeting criteria:**")
            st.dataframe(result.rules, use_container_width=True, hide_index=True)
        else:
            st.info("No association rules found with the specified thresholds. Try lowering minimum support or confidence.")

    with tab_items:
        if not result.itemsets.empty:
            st.markdown(f"**Found {len(result.itemsets)} frequent itemsets:**")
            st.dataframe(result.itemsets[["itemsets_str", "support", "length"]], use_container_width=True, hide_index=True)
        else:
            st.info("No frequent itemsets found with current minimum support.")

    with tab_scatter:
        if not result.rules.empty:
            fig = px.scatter(
                result.rules,
                x="support",
                y="confidence",
                color="lift",
                hover_data=["antecedents_str", "consequents_str"],
                title="Rule Distribution: Support vs Confidence (Colored by Lift)",
                color_continuous_scale="Plasma",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Generate rules to display scatter plot.")
